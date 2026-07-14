# Cascade plane-sweep Multi-View-Stereo cost-volume head.
#
# Given matching features for a reference view and N source views, together with
# the known metric camera transforms (cam_T_cam: ref->src) and intrinsics, this
# module builds a plane-sweep cost volume in *absolute metres*, regularizes it
# with a small 3D-conv U-Net, and reads out a metric depth map by soft-argmin.
#
# Two cascade stages are used:
#   1. coarse : uniform depth hypotheses over [min_depth, max_depth]
#   2. fine   : per-pixel hypotheses in a narrow window around the coarse depth
#
# Because the hypotheses are metric and the poses are metric, the output depth is
# metric - there is no scale ambiguity (this is what lets the model beat the
# monocular scale/shape floor on held-out stones).

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    g = min(max_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class _Conv3dGnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CostRegNet(nn.Module):
    """Small 3D-conv U-Net that regularizes a [B, C, D, h, w] cost volume.

    Returns a single-channel score volume [B, 1, D, h, w]; a softmax over the
    depth dimension turns it into a probability volume for soft-argmin.
    """

    def __init__(self, in_ch, base=8):
        super().__init__()
        self.conv0 = _Conv3dGnReLU(in_ch, base)
        self.conv1 = _Conv3dGnReLU(base, base * 2, stride=2)
        self.conv2 = _Conv3dGnReLU(base * 2, base * 2)
        self.conv3 = _Conv3dGnReLU(base * 2, base * 4, stride=2)
        self.conv4 = _Conv3dGnReLU(base * 4, base * 4)

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(base * 4, base * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
            _gn(base * 2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base * 2, base, 3, stride=2, padding=1, output_padding=1, bias=False),
            _gn(base), nn.ReLU(inplace=True))
        self.out = nn.Conv3d(base, 1, 3, padding=1)

    def forward(self, x):
        c0 = self.conv0(x)
        c2 = self.conv2(self.conv1(c0))
        c4 = self.conv4(self.conv3(c2))
        u = self.up1(c4)
        # Guard against off-by-one size mismatches from odd D/h/w.
        if u.shape[-3:] != c2.shape[-3:]:
            u = F.interpolate(u, size=c2.shape[-3:], mode="trilinear", align_corners=False)
        u = u + c2
        u = self.up2(u)
        if u.shape[-3:] != c0.shape[-3:]:
            u = F.interpolate(u, size=c0.shape[-3:], mode="trilinear", align_corners=False)
        u = u + c0
        return self.out(u)


def groupwise_correlation(fea_ref, fea_src, num_groups):
    """Group-wise correlation cost between two [B, C, h, w] feature maps.

    Splits channels into num_groups groups and averages the elementwise product
    within each group, yielding a [B, num_groups, h, w] cost that is far cheaper
    to regularize than a full feature-dim volume.
    """
    B, C, h, w = fea_ref.shape
    assert C % num_groups == 0, "feature channels must be divisible by num_groups"
    cpg = C // num_groups
    cost = (fea_ref * fea_src).view(B, num_groups, cpg, h, w).mean(dim=2)
    return cost


class PlaneSweepMVS(nn.Module):
    """Cascade plane-sweep MVS producing a metric depth map at feature resolution."""

    def __init__(self, feat_ch=32, num_groups=8, feat_scale=8,
                 min_depth=0.3, max_depth=1.0,
                 ndepth_coarse=48, ndepth_fine=48, fine_range_mm=20.0,
                 reg_base=8):
        super().__init__()
        self.feat_ch = feat_ch
        self.num_groups = num_groups
        self.feat_scale = float(feat_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.ndepth_coarse = int(ndepth_coarse)
        self.ndepth_fine = int(ndepth_fine)
        self.fine_range_m = float(fine_range_mm) / 1000.0

        self.reg_coarse = CostRegNet(num_groups, base=reg_base)
        self.reg_fine = CostRegNet(num_groups, base=reg_base)

    # ---- geometry helpers -------------------------------------------------
    def _scaled_K(self, K3x3):
        """Scale a full-resolution 3x3 K to feature resolution."""
        K = K3x3.clone().float()
        K[:, 0, :] = K[:, 0, :] / self.feat_scale  # fx, cx (and skew) along x
        K[:, 1, :] = K[:, 1, :] / self.feat_scale  # fy, cy along y
        return K

    @staticmethod
    def _pixel_grid(h, w, device, dtype):
        # Homogeneous pixel coordinates [3, h*w] without torch.meshgrid (which has
        # version-dependent `indexing` semantics).
        xs = torch.arange(w, device=device, dtype=dtype).view(1, w).expand(h, w)
        ys = torch.arange(h, device=device, dtype=dtype).view(h, 1).expand(h, w)
        ones = torch.ones(h, w, device=device, dtype=dtype)
        pix = torch.stack([xs, ys, ones], dim=0)  # [3, h, w]
        return pix.reshape(3, -1)  # [3, h*w]

    def _warp_src(self, feat_src, depth_map, T, K, invK):
        """Warp a source feature map into the reference frame at a given depth.

        feat_src : [B, C, h, w]  source features
        depth_map: [B, 1, h, w]  per-pixel reference depth hypothesis (metres)
        T        : [B, 4, 4]     cam_T_cam ref->src
        K, invK  : [B, 3, 3]     feature-resolution intrinsics and inverse
        returns    [B, C, h, w]  source features sampled at reference pixels
        """
        B, C, h, w = feat_src.shape
        device, dtype = feat_src.device, feat_src.dtype
        pix = self._pixel_grid(h, w, device, dtype).unsqueeze(0).expand(B, 3, h * w)
        cam = torch.bmm(invK, pix)                       # [B, 3, hw] unit rays
        cam = cam * depth_map.view(B, 1, h * w)          # scale by depth
        cam_h = torch.cat([cam, torch.ones(B, 1, h * w, device=device, dtype=dtype)], dim=1)
        P = torch.bmm(K, T[:, :3, :])                    # [B, 3, 4]
        proj = torch.bmm(P, cam_h)                       # [B, 3, hw]
        eps = 1e-7
        x = proj[:, 0] / (proj[:, 2] + eps)
        y = proj[:, 1] / (proj[:, 2] + eps)
        x = (x / (w - 1)) * 2.0 - 1.0
        y = (y / (h - 1)) * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).view(B, h, w, 2)
        return F.grid_sample(feat_src, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def _build_cost(self, feat_ref, feat_srcs, Ts, K, invK, depth_hyps):
        """Build the regularizable cost volume.

        depth_hyps: [B, D, h, w] per-pixel depth hypotheses (metres)
        returns cost volume [B, num_groups, D, h, w]
        """
        D = depth_hyps.shape[1]
        n_src = max(len(feat_srcs), 1)
        planes = []
        for j in range(D):
            dmap = depth_hyps[:, j:j + 1]  # [B, 1, h, w]
            acc = None
            for feat_src, T in zip(feat_srcs, Ts):
                warped = self._warp_src(feat_src, dmap, T, K, invK)
                corr = groupwise_correlation(feat_ref, warped, self.num_groups)
                acc = corr if acc is None else acc + corr
            planes.append(acc / n_src)  # [B, G, h, w]
        return torch.stack(planes, dim=2)  # [B, G, D, h, w]

    @staticmethod
    def _soft_argmin(score_vol, depth_hyps):
        """score_vol: [B, 1, D, h, w]; depth_hyps: [B, D, h, w] -> depth [B,1,h,w], prob."""
        prob = F.softmax(score_vol.squeeze(1), dim=1)      # [B, D, h, w]
        depth = torch.sum(prob * depth_hyps, dim=1, keepdim=True)
        return depth, prob

    def forward(self, feat_ref, feat_srcs, Ts, K3x3_full):
        """Run the two-stage cascade.

        feat_ref  : [B, C, h, w]
        feat_srcs : list of [B, C, h, w]
        Ts        : list of [B, 4, 4]  (ref->src), aligned with feat_srcs
        K3x3_full : [B, 3, 3] full-resolution intrinsics
        returns dict with 'depth' (fine, [B,1,h,w]) and 'depth_coarse'.
        """
        B, C, h, w = feat_ref.shape
        K = self._scaled_K(K3x3_full)
        invK = torch.inverse(K)

        # ---- coarse stage: uniform hypotheses over the full range ----
        dc = torch.linspace(self.min_depth, self.max_depth, self.ndepth_coarse,
                            device=feat_ref.device, dtype=feat_ref.dtype)
        depth_hyps_c = dc.view(1, self.ndepth_coarse, 1, 1).expand(B, self.ndepth_coarse, h, w)
        cost_c = self._build_cost(feat_ref, feat_srcs, Ts, K, invK, depth_hyps_c)
        score_c = self.reg_coarse(cost_c)
        depth_coarse, _ = self._soft_argmin(score_c, depth_hyps_c)

        # ---- fine stage: per-pixel window around the coarse estimate ----
        Df = self.ndepth_fine
        offs = torch.linspace(-self.fine_range_m, self.fine_range_m, Df,
                            device=feat_ref.device, dtype=feat_ref.dtype)
        depth_hyps_f = depth_coarse + offs.view(1, Df, 1, 1)  # [B, Df, h, w]
        depth_hyps_f = depth_hyps_f.clamp(self.min_depth, self.max_depth)
        cost_f = self._build_cost(feat_ref, feat_srcs, Ts, K, invK, depth_hyps_f)
        score_f = self.reg_fine(cost_f)
        depth_fine, _ = self._soft_argmin(score_f, depth_hyps_f)

        return {"depth": depth_fine, "depth_coarse": depth_coarse}
