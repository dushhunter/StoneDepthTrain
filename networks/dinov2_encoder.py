# Optional DINOv2 / Depth-Anything-V2 style encoder for the SPIdepth depth head.
#
# The default SPIdepth encoder is a timm CNN wrapped by networks/Unet.py. DINOv2
# vision transformers give much stronger, more generalizable dense features (they
# are the backbone behind Depth Anything V2), which is the biggest lever for
# generalizing depth to *unseen stone shapes*. A ViT does not fit the Unet skip
# structure, so this thin wrapper adapts it: run the ViT, reshape the patch tokens
# to a dense grid, project to `model_dim`, and upsample to ~1/`out_stride` input
# resolution so the query-transformer depth head can consume it unchanged.
#
# This path is strictly opt-in via --backbone dinov2_* and does not affect the
# default CNN encoder. Requires timm with DINOv2 weights available.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm import create_model
except Exception:  # pragma: no cover - timm always present in the training env
    create_model = None


# Friendly --backbone aliases -> timm model names.
_DINOV2_ALIASES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
}


def is_dinov2_backbone(name):
    return name in _DINOV2_ALIASES or name.startswith("vit_") and "dinov2" in name


class DINOv2Encoder(nn.Module):
    def __init__(self, backbone="dinov2_vits14", model_dim=32, pretrained=True, out_stride=4):
        super().__init__()
        if create_model is None:
            raise ImportError("timm is required for the DINOv2 encoder")
        model_name = _DINOV2_ALIASES.get(backbone, backbone)
        # num_classes=0 -> no classifier head; we use the patch tokens directly.
        self.vit = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch = self.vit.patch_embed.patch_size
        if isinstance(self.patch, (tuple, list)):
            self.patch = self.patch[0]
        self.num_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        embed = int(getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", 384)))
        self.out_stride = int(out_stride)
        self.proj = nn.Sequential(
            nn.Conv2d(embed, model_dim, kernel_size=1),
            nn.GroupNorm(min(8, model_dim), model_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
        )

    def _patch_tokens(self, x):
        """Return dense patch features [B, C, h, w] from the ViT."""
        feats = self.vit.forward_features(x)
        # timm ViTs return either [B, N, C] or a dict with 'x_norm_patchtokens'.
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"]
                prefix = 0
            else:
                tokens = feats.get("x", None)
                prefix = self.num_prefix
        else:
            tokens = feats
            prefix = self.num_prefix
        if prefix > 0:
            tokens = tokens[:, prefix:, :]
        return tokens

    def forward(self, x):
        B, C, H, W = x.shape
        # ViT needs input sizes that are a multiple of the patch size.
        newH = max(self.patch, (H // self.patch) * self.patch)
        newW = max(self.patch, (W // self.patch) * self.patch)
        xr = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
        tokens = self._patch_tokens(xr)                 # [B, h*w, embed]
        h, w = newH // self.patch, newW // self.patch
        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # [B, embed, h, w]
        feat = self.proj(feat)                          # [B, model_dim, h, w]
        out_h = max(1, H // self.out_stride)
        out_w = max(1, W // self.out_stride)
        return F.interpolate(feat, size=(out_h, out_w), mode="bilinear", align_corners=False)
