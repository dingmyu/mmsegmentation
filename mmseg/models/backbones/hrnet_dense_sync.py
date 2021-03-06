from mmcv.cnn import (build_norm_layer, constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.ops import resize

from ..builder import BACKBONES
import numbers
import collections
import logging
import functools
import torch
from torch import nn
from torch.nn import functional as F

norm_cfg = dict(type='SyncBN', requires_grad=True)

def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
    }[name]
    return active_fn

def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 expand_ratio=None,
                 kernel_sizes=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            build_norm_layer(norm_cfg, out_planes)[1],
            # nn.BatchNorm2d(out_planes, **batch_norm_kwargs),
            active_fn())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 kernel_size=3,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 expand_ratio=None,
                 kernel_sizes=None
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = None
        self.stride = stride
        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, planes)[1],
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_block_wrapper(block_str):
    """Wrapper for MobileNetV2 block.
    Use `expand_ratio` instead of manually specified channels number."""

    # return ConvBNReLU
    return BasicBlock


class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU')):
        super(ParallelModule, self).__init__()

        self.num_branches = num_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self._check_branches(
            num_branches, num_blocks, num_channels)
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)

    def _check_branches(self, num_branches, num_blocks, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x


class FuseModule(nn.Module):
    '''
        Consistant with HRNET:
        1. self.use_hr_format, eg: fuse 3 branches, and then add 4th branch from 3rd branch. (default fuse 4 branches)
        2. use_hr_format, if the channels are the same and stride==1, use None rather than fuse. (default, always fuse)
        3. self.in_channels_large_stride, use 16->16->64 instead of 16->32->64 for large stride. (default, True)
        4. The only difference in self.use_hr_format when adding a branch:
            is we use add 4th branch from 3rd branch, add 5th branch from 4rd branch
            hrnet use add 4th branch from 3rd branch, add 5th branch from 3rd branch (2 conv layers)
            actually only affect 1->2 stage
            can be hard coded: self.use_hr_format = self.use_hr_format and not(out_branches == 2 and in_branches == 1)
        5. hrnet have a fuse layer at the end, we remove it
    '''
    def __init__(self,
                 in_branches=1,
                 out_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 in_channels=[16],
                 out_channels=[16, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU'),
                 use_hr_format=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.in_channels_large_stride = True
        self.use_hr_format = use_hr_format and out_branches > in_branches  # w/o self, they are two different flags.
        self.use_hr_format = self.use_hr_format and not(out_branches == 2 and in_branches == 1)  # see 4.

        self.relu = nn.ReLU(False)
        if use_hr_format:
            block = ConvBNReLU

        fuse_layers = []
        for i in range(out_branches if not self.use_hr_format else in_branches):
            fuse_layer = []
            for j in range(in_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.active_fn,
                            kernel_size=1  # for hr format
                        ),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    if use_hr_format and in_channels[j] == out_channels[i]:
                        fuse_layer.append(None)
                    else:
                        fuse_layer.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=1,
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.active_fn,
                                kernel_size=3  # for hr format
                            ))
                else:
                    downsamples = []
                    for k in range(i - j):
                        if self.in_channels_large_stride:
                            if k == i - j - 1:
                                downsamples.append(
                                    block(
                                        in_channels[j],
                                        out_channels[i],
                                        expand_ratio=self.expand_ratio,
                                        kernel_sizes=self.kernel_sizes,
                                        stride=2,
                                        batch_norm_kwargs=self.batch_norm_kwargs,
                                        active_fn=self.active_fn,
                                        kernel_size=3  # for hr format
                                    ))
                            else:
                                downsamples.append(
                                    block(
                                        in_channels[j],
                                        in_channels[j],
                                        expand_ratio=self.expand_ratio,
                                        kernel_sizes=self.kernel_sizes,
                                        stride=2,
                                        batch_norm_kwargs=self.batch_norm_kwargs,
                                        active_fn=self.active_fn,
                                        kernel_size=3  # for hr format
                                    ))
                        else:
                            if k == 0:
                                downsamples.append(
                                    block(
                                        in_channels[j],
                                        out_channels[j + 1],
                                        expand_ratio=self.expand_ratio,
                                        kernel_sizes=self.kernel_sizes,
                                        stride=2,
                                        batch_norm_kwargs=self.batch_norm_kwargs,
                                        active_fn=self.active_fn,
                                        kernel_size=3  # for hr format
                                    ))
                            elif k == i - j - 1:
                                downsamples.append(
                                    block(
                                        out_channels[j + k],
                                        out_channels[i],
                                        expand_ratio=self.expand_ratio,
                                        kernel_sizes=self.kernel_sizes,
                                        stride=2,
                                        batch_norm_kwargs=self.batch_norm_kwargs,
                                        active_fn=self.active_fn,
                                        kernel_size=3  # for hr format
                                    ))
                            else:
                                downsamples.append(
                                    block(
                                        out_channels[j + k],
                                        out_channels[j + k + 1],
                                        expand_ratio=self.expand_ratio,
                                        kernel_sizes=self.kernel_sizes,
                                        stride=2,
                                        batch_norm_kwargs=self.batch_norm_kwargs,
                                        active_fn=self.active_fn,
                                        kernel_size=3  # for hr format
                                    ))
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        if self.use_hr_format:
            for branch in range(in_branches, out_branches):
                fuse_layers.append(nn.ModuleList([block(
                    out_channels[branch - 1],
                    out_channels[branch],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn,
                    kernel_size=3  # for hr format
                )]))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
            y = self.fuse_layers[i][0](x[0]) if self.fuse_layers[i][0] else x[0]  # hr_format, None
            for j in range(1, self.in_branches):
                if self.fuse_layers[i][j]:
                    y = y + self.fuse_layers[i][j](x[j])
                else:  # hr_format, None
                    y = y + x[j]
            x_fuse.append(self.relu(y))  # TODO(Mingyu): Use ReLU?
        if self.use_hr_format:
            for branch in range(self.in_branches, self.out_branches):
                x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        return x_fuse


@BACKBONES.register_module()
class HighResolutionNetSync(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_stride=4,
                 input_channel=16,
                 last_channel=1024,
                 head_channels=None,
                 bn_momentum=0.1,
                 bn_epsilon=1e-5,
                 dropout_ratio=0.2,
                 active_fn='nn.ReLU',
                 block='InvertedResidualChannels',
                 width_mult=1.0,
                 round_nearest=8,
                 expand_ratio=4,
                 kernel_sizes=[3, 5, 7],
                 inverted_residual_setting=None,
                 task='classification',
                 align_corners=False,
                 start_with_atomcell=False,
                 **kwargs):
        super(HighResolutionNetSync, self).__init__()

        batch_norm_kwargs = {
            'momentum': bn_momentum,
            'eps': bn_epsilon
        }

        self.avg_pool_size = input_size // 32
        self.input_stride = input_stride
        self.input_channel = _make_divisible(
            input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = get_active_fn(active_fn)
        self.kernel_sizes = kernel_sizes
        self.expand_ratio = expand_ratio
        self.task = task
        self.align_corners = align_corners

        self.block = get_block_wrapper(block)
        self.inverted_residual_setting = inverted_residual_setting

        downsamples = []
        if self.input_stride > 1:
            downsamples.append(ConvBNReLU(
                3,
                self.input_channel,
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        if self.input_stride > 2:
            downsamples.append(ConvBNReLU(
                self.input_channel,
                self.input_channel,
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        self.downsamples = nn.Sequential(*downsamples)

        features = []
        for index in range(len(inverted_residual_setting)):
            in_branches = 1 if index == 0 else inverted_residual_setting[index - 1][0]
            in_channels = [self.input_channel] if index == 0 else inverted_residual_setting[index - 1][-1]
            features.append(
                FuseModule(
                    in_branches=in_branches,
                    out_branches=inverted_residual_setting[index][0],
                    in_channels=in_channels,
                    out_channels=inverted_residual_setting[index][-1],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
            features.append(
                ParallelModule(
                    num_branches=inverted_residual_setting[index][0],
                    num_blocks=inverted_residual_setting[index][1],
                    num_channels=inverted_residual_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
        self.features = nn.Sequential(*features)
        self.init_weights()

    def init_weights(self, pretrained=None):
        # logging.info('=> init weights from normal distribution')
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


    def forward(self, x):
        x = self.downsamples(x)
        x = self.features([x])
        # x = self.classifier(x)
        return x
