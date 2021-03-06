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
import numpy as np
import math

checkpoint_kwparams = None
# checkpoint_kwparams = json.load(open('checkpoint.json'))


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.shape[0],) + x.shape[2:]).byte()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PosEncoder(nn.Module):
    def __init__(self, c=1, num_downsample=0, size=None):
        super(PosEncoder, self).__init__()

        self.size = size
        ds_conv = []

        for _ in range(num_downsample):
            ds_conv.append(nn.Conv2d(c,
                                     c,
                                     3,
                                     stride=2,
                                     padding=1,
                                     bias=False))
        ds_conv.append(nn.Conv2d(c, 1, 1))
        self.ds_conv = nn.Sequential(*ds_conv)

        if self.size:
            self.pos_dim = size[0] * size[1] // (4 ** num_downsample)
            self.pos_conv = nn.Conv1d(self.pos_dim, self.pos_dim, 1)

    def forward(self, token_coef):
        N, L, H, W = token_coef.shape
        token_coef = token_coef.reshape(N * L, 1, H, W)

        # interpolation to deal with input with varying sizes
        if self.size:
            token_coef = F.interpolate(token_coef, size=(self.size[0], self.size[1]))

        # downsampling
        token_coef = self.ds_conv(token_coef)
        token_coef = token_coef.view(N, L, -1).permute(0, 2, 1)

        # compress and compute the position encoding.
        if self.size:
            token_coef = self.pos_conv(token_coef)  # N, Cp, L
        return token_coef


class Tokenizer(nn.Module):
    '''
        l: number of tokenizers
        c: channels of feature map
        ct: channels of tokenizers
    '''

    def __init__(self, l, ct, c, head=3, groups=3):
        super(Tokenizer, self).__init__()

        # c -> l, get 2d attention score, It can be seen as a convolutional filter that divides the feature map X
        #   into various regions that corresponds to different semantic concepts.
        self.conv_token_coef = nn.Conv2d(c, l, kernel_size=1, padding=0, bias=False)

        # c -> c, get 2d feature (value), maybe not useful
        self.conv_value = nn.Conv2d(c, c, kernel_size=1, padding=0, bias=False, groups=groups)

        self.pos_encoding = PosEncoder(size=(16, 16), num_downsample=1)

        self.conv_token = nn.Conv1d(c + self.pos_encoding.pos_dim, ct, kernel_size=1, padding=0, bias=False)
        self.head = head
        self.c = c
        self.ct = ct

    # feature: N, C, H, W, token: N, CT , L
    def forward(self, feature):
        token_coef = self.conv_token_coef(feature)  # c -> l, get 2d tokens

        N, L, H, W = token_coef.shape
        token_coef = token_coef.view(N, 1, L, H * W)
        token_coef = token_coef.permute(0, 1, 3, 2)  # N, 1, HW, L
        token_coef = token_coef / np.sqrt(self.c)  # get stable gradient
        token_coef = F.softmax(token_coef, dim=2)  # get attention score along HW

        value = self.conv_value(feature).view(N, self.head, self.c // self.head, H * W)  # N, h, C//h, HW

        # (N, h, C//h, HW) * (N, 1, HW, L) -> N, h, C//h, L
        tokens = torch.matmul(value, token_coef).view(N, self.c, -1)  # N, C, L

        pos_encoding = self.pos_encoding(token_coef.permute(0, 3, 1, 2).reshape(N, L, H, W))  # N, cp (HW), L
        tokens = torch.cat((tokens, pos_encoding), dim=1)  # N, C + Cp (HW), L
        tokens = self.conv_token(tokens)  # N, Ct, L
        return tokens


class Transformer(nn.Module):
    def __init__(self, ct, head=3, kqv_groups=3):
        super(Transformer, self).__init__()
        self.k_conv = nn.Conv1d(ct, ct // 2, kernel_size=1, padding=0, bias=False, groups=kqv_groups)
        self.q_conv = nn.Conv1d(ct, ct // 2, kernel_size=1, padding=0, bias=False, groups=kqv_groups)
        self.v_conv = nn.Conv1d(ct, ct, kernel_size=1, padding=0, bias=False, groups=kqv_groups)
        self.ff_conv = nn.Conv1d(ct, ct, kernel_size=1, padding=0, bias=False)
        self.head = head
        self.ct = ct

    def forward(self, tokens):
        N = tokens.shape[0]  # # N, ct, l
        k = self.k_conv(tokens).view(N, self.head, self.ct // 2 // self.head, -1)  # N, h, ct // 2 // h, l
        q = self.q_conv(tokens).view(N, self.head, self.ct // 2 // self.head, -1)  # N, h, ct // 2 // h, l
        v = self.v_conv(tokens).view(N, self.head, self.ct // self.head, -1)  # N, h, ct // h, l

        # (N, h, l, ct // 2 // h) * (N, h, ct // 2 // h, l) -> N, h, l, l
        kq = torch.matmul(k.permute(0, 1, 3, 2), q)
        kq = F.softmax(kq / np.sqrt(kq.shape[2]), dim=2)

        # (N, h, ct // h, l) * (N, h, l, l) -> N, h, ct // h, l
        kqv = torch.matmul(v, kq).view(N, self.ct, -1)  # N, ct, l
        tokens = tokens + kqv
        tokens = tokens + self.ff_conv(tokens)  # Maybe useless
        return tokens  # N, ct, l


class Projector(nn.Module):
    def __init__(self, CT, C, head=3, groups=3):
        super(Projector, self).__init__()
        self.head = head

        self.proj_value_conv = nn.Conv1d(CT, C, 1)
        self.proj_key_conv = nn.Conv1d(CT, C, 1)
        # self.proj_query_conv = nn.Conv2d(C, CT, 1, groups=groups)

    def forward(self, feature, token):
        N, _, L = token.shape  # N, ct, l.  feature: N, C, H, W.
        h = self.head

        proj_v = self.proj_value_conv(token).view(N, h, -1, L)  # N, h, c/h, l
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)  # N, h, c/h, l
        proj_q = feature
        # proj_q = self.proj_query_conv(feature)   # N, ct, H, W

        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)  # （N, h, C // h, HW) -> （N, h, HW, C // h）
        proj_coef = F.softmax(torch.matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)  # N, h, HW , L
        proj = torch.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))  # N, h, C//h, HW
        _, _, H, W = feature.shape
        proj = proj.view(N, -1, H, W)
        return feature + proj.view(N, -1, H, W)


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
        # assert stride in [1, 2]
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


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


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
                 dilation=1,
                 padding=None,
                 **kwargs):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        if not padding:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      dilation=dilation,
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = active_fn()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        self.stride = stride
        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
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

    if block_str == 'InvertedResidualChannels':
        return InvertedResidual
    elif block_str == 'ConvBNReLU':
        return ConvBNReLU
    elif block_str == 'BasicBlock':
        return BasicBlock
    else:
        raise ValueError('Unknown type of blocks.')



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
                 use_hr_format=False,
                 only_fuse_neighbor=True,
                 directly_downsample=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.only_fuse_neighbor = only_fuse_neighbor
        self.in_channels_large_stride = True  # see 3.
        if only_fuse_neighbor:
            self.use_hr_format = out_branches > in_branches
            # w/o self, are two different flags. (see 1.)
        else:
            self.use_hr_format = out_branches > in_branches and \
                                 not (out_branches == 2 and in_branches == 1)  # see 4.

        self.relu = functools.partial(nn.ReLU, inplace=False)
        if use_hr_format:
            block = ConvBNReLU  # See 2.
        block = ConvBNReLU

        fuse_layers = []
        for i in range(out_branches if not self.use_hr_format else in_branches):
            fuse_layer = []
            for j in range(in_branches):
                if only_fuse_neighbor:
                    if j < i - 1 or j > i + 1:
                        fuse_layer.append(None)
                        continue
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.relu if not use_hr_format else None,
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
                                active_fn=self.relu if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                else:
                    downsamples = []
                    if directly_downsample:
                        downsamples.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=2 ** (i - j),
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.relu if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                    else:
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
                                            active_fn=self.relu if not use_hr_format else None,
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
                                            active_fn=self.relu,
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
                                            active_fn=self.relu if not (use_hr_format and i == j + 1) else None,
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
                                            active_fn=self.relu if not use_hr_format else None,
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
                                            active_fn=self.relu,
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
                    active_fn=self.relu,
                    kernel_size=3  # for hr format
                )]))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        if not self.only_fuse_neighbor:
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
        else:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                flag = 1
                for j in range(i-1, i+2):
                    if 0 <= j < self.in_branches:
                        if flag:
                            y = self.fuse_layers[i][j](x[j]) if self.fuse_layers[i][j] else x[j]  # hr_format, None
                            flag = 0
                        else:
                            if self.fuse_layers[i][j]:
                                y = y + resize(
                                    self.fuse_layers[i][j](x[j]),
                                    size=y.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
                            else:  # hr_format, None
                                y = y + x[j]
                x_fuse.append(self.relu()(y))  # TODO(Mingyu): Use ReLU?
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
                 input_channel=[16, 16],
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
        self.input_channel = [_make_divisible(item * width_mult, round_nearest) for item in input_channel]
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
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
                input_channel[0],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        if self.input_stride > 2:
            if start_with_atomcell:
                downsamples.append(InvertedResidual(input_channel[0],
                                                    input_channel[0],
                                                    1,
                                                    1,
                                                    [3],
                                                    self.active_fn,
                                                    self.batch_norm_kwargs))
            downsamples.append(ConvBNReLU(
                input_channel[0],
                input_channel[1],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        self.downsamples = nn.Sequential(*downsamples)

        features = []
        for index in range(len(inverted_residual_setting)):
            in_branches = 1 if index == 0 else inverted_residual_setting[index - 1][0]
            in_channels = [input_channel[1]] if index == 0 else inverted_residual_setting[index - 1][-1]
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
                last_channel=last_channel,
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
                    last_channel,
                    kernel_size=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn
                )
            else:
                self.transform = self.block(
                        sum(inverted_residual_setting[-1][-1]),
                        last_channel,
                        expand_ratio=self.expand_ratio,
                        kernel_sizes=self.kernel_sizes,
                        stride=1,
                        batch_norm_kwargs=self.batch_norm_kwargs,
                        active_fn=self.active_fn,
                    )
            self.classifier = nn.Conv2d(last_channel,
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
