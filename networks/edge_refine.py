from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeRefine(nn.Module):
    """RGB-guided full-resolution depth refinement head.

    The depth decoder predicts metric depth at a reduced spatial resolution
    (half the input) and it is a soft expectation over bins, so stone
    silhouettes come out as rounded ramps once bilinearly upsampled. This head
    fuses the coarse depth with sharp full-resolution RGB features and predicts
    a residual, transferring the crisp image edge onto the depth.

    The final convolution is initialised near zero so the module starts as an
    identity (behaves like the previous bilinear upsampling) and only learns to
    add edge detail, which keeps early training stable.
    """

    def __init__(self, base_channels=32, min_val=0.30, max_val=1.00,
                 groups=8):
        super(EdgeRefine, self).__init__()
        gn = min(groups, base_channels)

        # Guidance branch: shallow features from the full-resolution RGB image.
        self.rgb = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn, base_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion branch: coarse depth (1ch) + rgb features -> residual (1ch).
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn, base_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        self.min_val = min_val
        self.max_val = max_val

        # Start as identity: no residual, output == bilinear-upsampled coarse.
        nn.init.zeros_(self.residual.weight)
        if self.residual.bias is not None:
            nn.init.zeros_(self.residual.bias)

    def forward(self, coarse_depth, rgb):
        """coarse_depth: [B,1,h,w] metric depth. rgb: [B,3,H,W] image.

        Returns sharpened metric depth at the RGB (full input) resolution.
        """
        H, W = rgb.shape[-2:]
        if coarse_depth.shape[-2:] != (H, W):
            coarse_up = F.interpolate(
                coarse_depth, size=(H, W), mode="bilinear", align_corners=False)
        else:
            coarse_up = coarse_depth
        g = self.rgb(rgb)
        res = self.residual(self.fuse(torch.cat([coarse_up, g], dim=1)))
        out = coarse_up + res
        return out.clamp(self.min_val, self.max_val)
