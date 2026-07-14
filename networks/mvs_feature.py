# Shared lightweight FPN feature extractor for the plane-sweep MVS head.
#
# It maps an input RGB view [B, 3, H, W] to a dense feature map
# [B, out_ch, H/feat_scale, W/feat_scale]. The same network is applied to the
# reference view and every source view, so the cost volume matches features that
# were produced by identical weights (Siamese matching). GroupNorm is used
# instead of BatchNorm so the network is stable with the tiny batch sizes typical
# of this turntable dataset.

import torch.nn as nn


def _gn(num_channels, max_groups=8):
    """Pick a GroupNorm group count that divides num_channels."""
    g = min(max_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class _ConvGnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel=3, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MVSFeatureNet(nn.Module):
    """Small strided-conv FPN producing matching features at 1/feat_scale res.

    feat_scale must be one of {2, 4, 8}. The network downsamples by a factor of 2
    per stage; the requested feat_scale selects how many stages are used.
    """

    def __init__(self, out_ch=32, base_ch=16, feat_scale=8):
        super().__init__()
        assert feat_scale in (2, 4, 8), "feat_scale must be 2, 4 or 8"
        self.feat_scale = feat_scale

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Stage 0 keeps resolution, stages 1..3 halve it.
        self.stem = nn.Sequential(_ConvGnReLU(3, c1), _ConvGnReLU(c1, c1))
        self.down1 = nn.Sequential(_ConvGnReLU(c1, c2, stride=2), _ConvGnReLU(c2, c2))   # 1/2
        self.down2 = nn.Sequential(_ConvGnReLU(c2, c3, stride=2), _ConvGnReLU(c3, c3))   # 1/4
        self.down3 = nn.Sequential(_ConvGnReLU(c3, c3, stride=2), _ConvGnReLU(c3, c3))   # 1/8

        self.head = nn.Conv2d(c3, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        if self.feat_scale >= 4:
            x = self.down2(x)
        if self.feat_scale >= 8:
            x = self.down3(x)
        return self.head(x)
