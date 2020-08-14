from ..builder import BACKBONES
import numbers
import collections
import logging
import functools
import torch
from torch import nn
from torch.nn import functional as F

from mmseg.ops import resize
import json

checkpoint_kwparams = None
checkpoint_kwparams = json.load(open('checkpoint.json'))


class InvertedResidualChannels(nn.Module):
    """MobiletNetV2 building block."""

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None):
        super(InvertedResidualChannels, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn

        self.ops, self.pw_bn = self._build(channels, kernel_sizes, expand)

        if not self.use_res_connect:  # TODO(Mingyu): Add this residual
            # assert (self.input_dim % min(self.input_dim, self.output_dim) == 0
            #         and self.output_dim % min(self.input_dim, self.output_dim) == 0)
            group = [x for x in range(1, self.input_dim + 1)
                     if self.input_dim % x == 0 and self.output_dim % x == 0][-1]
            self.residual = nn.Conv2d(self.input_dim,
                                      self.output_dim,
                                      kernel_size=1,
                                      stride=self.stride,
                                      padding=0,
                                      groups=group,
                                      bias=False)

    def _build(self, hidden_dims, kernel_sizes, expand):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        narrow_start = 0
        ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                # pw
                layers.append(
                    ConvBNReLU(self.input_dim,
                               hidden_dim,
                               kernel_size=1,
                               batch_norm_kwargs=_batch_norm_kwargs,
                               active_fn=self.active_fn))
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                logging.warning(
                    'uncomment this for previous trained search_first model')
                # layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
                # pw-linear
                nn.Conv2d(hidden_dim, self.output_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup, **batch_norm_kwargs),
            ])
            ops.append(nn.Sequential(*layers))
        pw_bn = None
        if len(ops) != 0:
            pw_bn = nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs)

        if not expand and narrow_start != self.input_dim:
            raise ValueError('Part of input are not used')

        return ops, pw_bn


    def forward(self, x):
        # logging.warning(
        #     'The whole block is pruned')
        if len(self.ops) == 0:
            if not self.use_res_connect:
                return self.residual(x)
            else:
                return x
        tmp = sum([op(x) for op in self.ops])
        tmp = self.pw_bn(tmp)
        if self.use_res_connect:
            return x + tmp
        else:
            return self.residual(x) + tmp
        return tmp

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={},'
                ' stride={})').format(self._get_name(), self.input_dim,
                                      self.output_dim, self.channels,
                                      self.kernel_sizes, self.expand,
                                      self.stride)


class InvertedResidual(InvertedResidualChannels):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 kernel_sizes,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 **kwargs):

        def _expand_ratio_to_hiddens(expand_ratio):
            if isinstance(expand_ratio, list):
                assert len(expand_ratio) == len(kernel_sizes)
                expand = True
            elif isinstance(expand_ratio, numbers.Number):
                expand = expand_ratio != 1
                expand_ratio = [expand_ratio for _ in kernel_sizes]
            else:
                raise ValueError(
                    'Unknown expand_ratio type: {}'.format(expand_ratio))
            hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            return hidden_dims, expand

        hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
        if checkpoint_kwparams:
            assert oup == checkpoint_kwparams[0][0]
            print('loading: {} -> {}, {} -> {}'.format(
                hidden_dims, checkpoint_kwparams[0][4], kernel_sizes, checkpoint_kwparams[0][3]))
            hidden_dims = checkpoint_kwparams[0][4]
            kernel_sizes = checkpoint_kwparams[0][3]
            checkpoint_kwparams.pop(0)

        super(InvertedResidual,
              self).__init__(inp,
                             oup,
                             stride,
                             hidden_dims,
                             kernel_sizes,
                             expand,
                             active_fn=active_fn,
                             batch_norm_kwargs=batch_norm_kwargs)
        self.expand_ratio = expand_ratio


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


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
                 **kwargs):
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
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn() if active_fn is not None else Identity())


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
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = None
        self.stride = stride
        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=0.1),
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
    return InvertedResidual


class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
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
        Consistent with HRNET:
        1. self.use_hr_format, eg: fuse 3 branches, and then add 4th branch from 3rd branch. (default fuse 4 branches)
        2. use_hr_format, if the channels are the same and stride==1, use None rather than fuse. (default, always fuse)
            and use convbnrelu, and kernel_size=1 when upsample.
            also control the relu here (last layer no relu)
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
                 active_fn=get_active_fn('nn.ReLU6'),
                 use_hr_format=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.in_channels_large_stride = True  # see 3.
        self.use_hr_format = use_hr_format and out_branches > in_branches  # w/o self, are two different flags. (see 1.)
        self.use_hr_format = self.use_hr_format and not(out_branches == 2 and in_branches == 1)  # see 4.
        use_hr_format = False

        self.relu = self.active_fn()
        if use_hr_format:
            block = ConvBNReLU  # See 2.

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
                            active_fn=self.active_fn if not use_hr_format else None,
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
                                active_fn=self.active_fn if not use_hr_format else None,
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
                                        active_fn=self.active_fn if not use_hr_format else None,
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
                                        active_fn=self.active_fn if not (use_hr_format and i == j + 1) else None,
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
                                        active_fn=self.active_fn if not use_hr_format else None,
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
class HighResolutionNet(nn.Module):

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
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 width_mult=1.0,
                 round_nearest=8,
                 expand_ratio=4,
                 kernel_sizes=[3, 5, 7],
                 inverted_residual_setting=None,
                 task='segmentation',
                 align_corners=False,
                 start_with_atomcell=False,
                 fcn_head_for_seg=False,
                 **kwargs):
        super(HighResolutionNet, self).__init__()

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
            if start_with_atomcell:
                downsamples.append(InvertedResidual(self.input_channel,
                                                    self.input_channel,
                                                    1,
                                                    1,
                                                    [3],
                                                    self.active_fn,
                                                    self.batch_norm_kwargs))
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

        if self.task == 'classification':
            features.append(HeadModule(
                pre_stage_channels=inverted_residual_setting[-1][2],
                head_channels=head_channels,
                last_channel=self.last_channel,
                avg_pool_size=self.avg_pool_size,
                block=self.block,
                expand_ratio=self.expand_ratio,
                kernel_sizes=self.kernel_sizes,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_ratio),
                nn.Linear(last_channel, num_classes),
            )
        elif self.task == 'segmentation':
            if fcn_head_for_seg:
                self.transform = ConvBNReLU(
                    sum(inverted_residual_setting[-1][-1]),
                    self.last_channel,
                    kernel_size=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn
                )
            else:
                self.transform = self.block(
                        sum(inverted_residual_setting[-1][-1]),
                        self.last_channel,
                        expand_ratio=2,
                        kernel_sizes=self.kernel_sizes,
                        stride=1,
                        batch_norm_kwargs=self.batch_norm_kwargs,
                        active_fn=self.active_fn,
                    )
            self.classifier = nn.Conv2d(self.last_channel,
                                        num_classes,
                                        kernel_size=1)

        self.features = nn.Sequential(*features)
        self.init_weights()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        inputs = self.transform(inputs)
        return inputs

    def init_weights(self, pretrained=None):
        logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsamples(x)
        x = self.features([x])
        if self.task == 'segmentation':
            x = self._transform_inputs(x)
        x = self.classifier(x)
        return x
