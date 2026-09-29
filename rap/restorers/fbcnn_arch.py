"""FBCNN architecture (Jiang et al., ICCV 2021), split into encode/decode so the
decoder can be rerun for different quality factors.

Building blocks are copied from https://github.com/jiaxi-jiang/FBCNN (Apache-2.0), whose
blocks come from KAIR (https://github.com/cszn/KAIR, MIT, Copyright (c) 2019 Kai Zhang).
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)



class QFAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(QFAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res(x) + beta
        return x + res


class FBCNN(nn.Module):
    """U-Net whose decoder is modulated by a quality/degradation code.

    ``forward`` returns (restored, predicted_code); passing ``cond`` overrides
    the predicted code (1 - QF/100), which is what makes the output
    controllable.
    """

    def __init__(self, in_nc=3, out_nc=3, nc=(64, 128, 256, 512), nb=4, act_mode='R'):
        super().__init__()
        self.nb = nb
        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')

        def res_blocks(ch):
            return [ResBlock(ch, ch, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]

        self.m_down1 = sequential(*res_blocks(nc[0]), downsample_strideconv(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(*res_blocks(nc[1]), downsample_strideconv(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(*res_blocks(nc[2]), downsample_strideconv(nc[2], nc[3], bias=True, mode='2'))
        self.m_body_encoder = sequential(*res_blocks(nc[3]))
        self.m_body_decoder = sequential(*res_blocks(nc[3]))

        def up(cin, cout):
            return nn.ModuleList([upsample_convtranspose(cin, cout, bias=True, mode='2'),
                                  *[QFAttention(cout, cout, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up3 = up(nc[3], nc[2])
        self.m_up2 = up(nc[2], nc[1])
        self.m_up1 = up(nc[1], nc[0])
        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')

        self.qf_pred = sequential(*res_blocks(nc[3]),
                                  nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                  nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, 1), nn.Sigmoid())
        self.qf_embed = sequential(nn.Linear(1, 512), nn.ReLU(),
                                   nn.Linear(512, 512), nn.ReLU(),
                                   nn.Linear(512, 512), nn.ReLU())
        self.to_gamma_3 = sequential(nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = sequential(nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = sequential(nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = sequential(nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = sequential(nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = sequential(nn.Linear(512, nc[0]), nn.Tanh())

    def encode(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % 8, 0, -h % 8), mode='replicate')
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        z = self.m_body_encoder(x4)
        return (x1, x2, x3, x4, z), self.qf_pred(z), (h, w)

    def decode(self, feats, code, size):
        x1, x2, x3, x4, z = feats
        e = self.qf_embed(code)
        x = self.m_body_decoder(z) + x4
        for ups, skip, g, b in ((self.m_up3, x3, self.to_gamma_3, self.to_beta_3),
                                (self.m_up2, x2, self.to_gamma_2, self.to_beta_2),
                                (self.m_up1, x1, self.to_gamma_1, self.to_beta_1)):
            x = ups[0](x)
            gamma, beta = g(e), b(e)
            for blk in ups[1:]:
                x = blk(x, gamma, beta)
            x = x + skip
        x = self.m_tail(x)
        return x[..., :size[0], :size[1]]

    def forward(self, x, cond=None):
        feats, code, size = self.encode(x)
        return self.decode(feats, code if cond is None else cond, size), code
