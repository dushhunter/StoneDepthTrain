# Inference for the cost-volume MVS depth head.
#
# Runs the trained plane-sweep MVS network on the turntable views of a single
# stone and saves a metric depth map per reference frame. Depth is triangulated
# from the known-pose neighbour views, so the output is metric (millimetres) with
# no scale ambiguity - no plane anchoring needed.
#
# Example:
#   python3 test_mvs.py @configs/infer_mvs_args.txt
#
# The config/CLI must provide (reusing MonodepthOptions):
#   --image_path <folder of NNNN.png for one stone>
#   --intrinsics_file_path <KV intrinsics file>   (same as training)
#   --load_weights_folder <folder with mvs_feature.pth + mvs.pth>
#   --use_known_pose --use_mvs --height --width --min_depth --max_depth
#   --mvs_frame_offsets -8 -4 4 8   (source views)
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import math
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.cm as cm
import torch
from torchvision import transforms

import networks
from SQLdepth import MonodepthOptions


def _read_intrinsics(file_name, folder_key, width, height):
    """Return (K3x3 float tensor [3,3] at pixel res, axis_depth or None)."""
    K = None
    axis_depth = None
    with open(file_name, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[0] != folder_key:
                continue
            fx, fy, cx, cy = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
            Kn = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            Kn[0, :] *= width
            Kn[1, :] *= height
            K = torch.from_numpy(Kn)
            if len(parts) >= 6:
                try:
                    axis_depth = float(parts[5])
                except ValueError:
                    axis_depth = None
            break
    if K is None:
        raise KeyError("folder '{}' not found in {}".format(folder_key, file_name))
    return K, axis_depth


def _turntable_transform(frame_offset, angle_deg, axis_depth, axis_offset_x, device):
    """4x4 rigid transform ref->src for a turntable offset (X' = R X + (I-R) c)."""
    a = frame_offset * angle_deg * math.pi / 180.0
    ca, sa = math.cos(a), math.sin(a)
    R = torch.tensor([[ca, 0.0, sa, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [-sa, 0.0, ca, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    c = torch.tensor([axis_offset_x, 0.0, axis_depth], dtype=torch.float32, device=device)
    R3 = R[:3, :3]
    t = (torch.eye(3, dtype=torch.float32, device=device) - R3) @ c
    T = R.clone()
    T[:3, 3] = t
    return T.unsqueeze(0)  # [1,4,4]


def _load_color(path, width, height, to_tensor, device):
    img = pil.open(path).convert("RGB").resize((width, height), pil.BILINEAR)
    return to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]


def _strip_module(state):
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(opt, "no_cuda", False) else "cpu")
    assert opt.use_known_pose, "test_mvs requires --use_known_pose"

    folder = os.path.normpath(opt.image_path)
    folder_key = os.path.basename(folder)
    frames_per_seq = int(getattr(opt, "frames_per_seq", 120))

    K3x3, axis_depth_file = _read_intrinsics(
        opt.intrinsics_file_path, folder_key, opt.width, opt.height)
    K3x3 = K3x3.unsqueeze(0).to(device)  # [1,3,3]

    if getattr(opt, "turntable_axis_depth", -1.0) > 0:
        axis_depth = float(opt.turntable_axis_depth)
    elif axis_depth_file is not None:
        axis_depth = axis_depth_file
    else:
        axis_depth = 0.5 * (opt.min_depth + opt.max_depth)
        print("   [warn] no metric axis depth given; using range midpoint {:.3f}m "
              "(scale may be off)".format(axis_depth))

    feat_scale = getattr(opt, "mvs_feat_scale", 8)
    feat_ch = getattr(opt, "mvs_feature_ch", 32)
    feat_net = networks.MVSFeatureNet(out_ch=feat_ch, feat_scale=feat_scale).to(device)
    mvs = networks.PlaneSweepMVS(
        feat_ch=feat_ch, num_groups=getattr(opt, "mvs_num_groups", 8),
        feat_scale=feat_scale, min_depth=opt.min_depth, max_depth=opt.max_depth,
        ndepth_coarse=getattr(opt, "mvs_num_depth_coarse", 48),
        ndepth_fine=getattr(opt, "mvs_num_depth_fine", 48),
        fine_range_mm=getattr(opt, "mvs_fine_range_mm", 20.0)).to(device)

    feat_net.load_state_dict(_strip_module(torch.load(
        os.path.join(opt.load_weights_folder, "mvs_feature.pth"), map_location=device)))
    mvs.load_state_dict(_strip_module(torch.load(
        os.path.join(opt.load_weights_folder, "mvs.pth"), map_location=device)))
    feat_net.eval()
    mvs.eval()

    to_tensor = transforms.ToTensor()
    offsets = list(getattr(opt, "mvs_frame_offsets", []) or [-8, -4, 4, 8])

    out_dir = os.path.join(folder, "mvs_out")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(folder, "[0-9][0-9][0-9][0-9].png")))
    print("-> {} reference frames, sources {}, axis_depth={:.4f}m".format(
        len(paths), offsets, axis_depth))

    Ts = [_turntable_transform(off, opt.turntable_angle_deg, axis_depth,
                               opt.turntable_axis_offset_x, device) for off in offsets]

    with torch.no_grad():
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            ref_idx = int(name)
            ref = _load_color(p, opt.width, opt.height, to_tensor, device)
            feat_ref = feat_net(ref)
            feat_srcs = []
            for off in offsets:
                sidx = ((ref_idx - 1 + off) % frames_per_seq) + 1
                sp = os.path.join(folder, "{:04d}.png".format(sidx))
                feat_srcs.append(feat_net(_load_color(sp, opt.width, opt.height, to_tensor, device)))

            out = mvs(feat_ref, feat_srcs, Ts, K3x3)
            depth = torch.nn.functional.interpolate(
                out["depth"], size=(opt.height, opt.width),
                mode="bilinear", align_corners=False)
            depth_np = depth.squeeze().cpu().numpy().astype(np.float32)  # metres

            # Metric uint16 in millimetres (lossless, matches training GT scale).
            mm = np.clip(depth_np * 1000.0, 0, 65535).astype("uint16")
            pil.fromarray(mm).save(os.path.join(out_dir, "{}_depth_mm.png".format(name)))
            np.save(os.path.join(out_dir, "{}_depth.npy".format(name)), depth_np)

            # Colourised preview for quick visual inspection.
            vmin = float(np.percentile(depth_np, 5))
            vmax = float(np.percentile(depth_np, 95))
            norm = np.clip((depth_np - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            colored = (cm.get_cmap("magma")(norm)[:, :, :3] * 255).astype("uint8")
            pil.fromarray(colored).save(os.path.join(out_dir, "{}_depth.jpeg".format(name)))
            print("   {} -> depth [{:.3f}, {:.3f}] m".format(
                name, float(depth_np.min()), float(depth_np.max())))

    print("-> saved metric depth to", out_dir)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        opt = options.parser.parse_args(["@" + sys.argv[1]])
    else:
        opt = options.parser.parse_args()
    main(opt)
