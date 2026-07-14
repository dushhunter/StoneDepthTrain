"""Standalone SPIdepth-style depth evaluation for the stone dataset.

Loads a trained SQLdepth checkpoint and evaluates it over a split file
(default: the validation split), reporting the 7 standard metrics
(AbsRel, SqRel, RMSE, RMSElog, delta1/2/3) using the same per-image
median-scaling protocol SPIdepth uses. Produces four variants:

    img_ms / img_abs     : whole valid image, median-scaled / absolute
    stone_ms / stone_abs : stone region,     median-scaled / absolute

This mirrors trainer.validate_full_epoch but runs independently from a
weights folder, so it is a clean, re-runnable artifact for the thesis.

Usage:
    python3 evaluate_stone.py ./configs/infer_args.txt \
        --data_path <dataset_root> \
        --gt_depth_path <dataset_root> \
        --gt_depth_subdir data_depth_annotated/train/groundtruth_float32png \
        --split stone
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

from SQLdepth import MonodepthOptions, SQLdepth
from layers import compute_depth_errors

METRIC7 = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
VARIANTS = ["img_ms", "img_abs", "stone_ms", "stone_abs"]


def _decode_float32_rgba_depth(depth_img):
    rgba = np.array(depth_img.convert("RGBA"), dtype=np.uint8)
    h, w, c = rgba.shape
    if c != 4:
        raise ValueError("Expected RGBA depth image with 4 channels")
    return rgba.reshape(-1, 4).view("<f4").reshape(h, w).copy()


def _load_gt_depth(path, encoding, scale):
    dp = pil.open(path)
    if encoding == "uint16":
        return np.array(dp, dtype=np.float32) / scale
    if encoding == "float32_rgba" or dp.mode == "RGBA":
        return _decode_float32_rgba_depth(dp).astype(np.float32)
    return np.array(dp, dtype=np.float32) / scale


def _suite(pred_t, gt_t, region_mask, scale, min_val, max_val, min_pixels=64):
    """7 metrics on region_mask after pred*scale (clamped). Returns np[7] or None."""
    if region_mask.sum().item() < min_pixels:
        return None
    gt_sel = gt_t[region_mask]
    pred_sel = (pred_t[region_mask] * float(scale)).clamp(min_val, max_val)
    errs = compute_depth_errors(gt_sel, pred_sel)
    return np.array([float(e.item()) for e in errs], dtype=np.float64)


def evaluate(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda)
                          else "cpu")
    model = SQLdepth(args)
    model.to(device)
    model.eval()

    split_file = args.eval_split_file or os.path.join(
        "splits", args.split, "val_files.txt")
    with open(split_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print("-> Evaluating on {} samples from {}".format(len(lines), split_file))

    use_ms = not args.disable_median_scaling
    suite_sum = {k: np.zeros(7, dtype=np.float64) for k in VARIANTS}
    suite_cnt = {k: 0 for k in VARIANTS}

    with torch.no_grad():
        for idx, line in enumerate(lines):
            parts = line.split()
            folder = parts[0]
            frame = int(parts[1]) if len(parts) > 1 else 0

            color_path = os.path.join(args.data_path, folder, "{:04d}.png".format(frame))
            gt_path = os.path.join(args.gt_depth_path, args.gt_depth_subdir, folder,
                                   "depth_{:04d}.png".format(frame))
            mask_path = os.path.join(args.data_path, folder, args.mask_subdir,
                                     "mask_{:04d}.png".format(frame))
            if not (os.path.isfile(color_path) and os.path.isfile(gt_path)):
                print("   skip (missing): {}".format(color_path))
                continue

            img = pil.open(color_path).convert("RGB")
            img = img.resize((args.width, args.height), pil.LANCZOS)
            img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)

            pred = model(img_t)
            pred_flipped = torch.flip(model(torch.flip(img_t, dims=[-1])), dims=[-1])
            pred = (pred + pred_flipped) / 2.0

            gt_np = _load_gt_depth(gt_path, args.gt_depth_encoding, args.gt_depth_scale)
            gt_t = torch.from_numpy(gt_np).to(device)
            gh, gw = gt_t.shape

            if pred.shape[-2:] != (gh, gw):
                pred = torch.nn.functional.interpolate(
                    pred, (gh, gw), mode="bilinear", align_corners=False)
            pred_t = torch.clamp(pred.squeeze(), args.min_depth, args.max_depth)

            valid = (gt_t > args.min_depth) & (gt_t < args.max_depth)
            if valid.sum().item() == 0:
                continue

            s_img = (torch.median(gt_t[valid]) /
                     torch.median(pred_t[valid]).clamp(min=1e-6)).item()
            if args.pred_depth_scale_factor and args.pred_depth_scale_factor != 1.0:
                s_img = float(args.pred_depth_scale_factor)

            m = _suite(pred_t, gt_t, valid, 1.0, args.min_depth, args.max_depth)
            if m is not None:
                suite_sum["img_abs"] += m
                suite_cnt["img_abs"] += 1
            if use_ms:
                m = _suite(pred_t, gt_t, valid, s_img, args.min_depth, args.max_depth)
                if m is not None:
                    suite_sum["img_ms"] += m
                    suite_cnt["img_ms"] += 1

            if os.path.isfile(mask_path):
                mk = pil.open(mask_path).convert("L").resize((gw, gh), pil.NEAREST)
                stone = valid & (torch.from_numpy(np.array(mk)).to(device) > 127)
                m = _suite(pred_t, gt_t, stone, 1.0, args.min_depth, args.max_depth)
                if m is not None:
                    suite_sum["stone_abs"] += m
                    suite_cnt["stone_abs"] += 1
                if use_ms:
                    m = _suite(pred_t, gt_t, stone, s_img, args.min_depth, args.max_depth)
                    if m is not None:
                        suite_sum["stone_ms"] += m
                        suite_cnt["stone_ms"] += 1

            if (idx + 1) % 20 == 0:
                print("   processed {}/{}".format(idx + 1, len(lines)))

    suite_avg = {k: (suite_sum[k] / suite_cnt[k]) if suite_cnt[k] > 0 else None
                 for k in VARIANTS}
    _print_table(suite_avg)
    _write_csv(args.eval_out, suite_avg)
    return suite_avg


def _print_table(suite_avg):
    labels = [("img_ms", "img  (scaled)"), ("img_abs", "img  (abs)   "),
              ("stone_ms", "stone(scaled)"), ("stone_abs", "stone(abs)   ")]
    print("\nDepth metrics (per-image mean; 'scaled' = median-scaled, SPIdepth protocol):")
    print("                 AbsRel   SqRel    RMSE     RMSElog  d1      d2      d3")
    for key, label in labels:
        vals = suite_avg.get(key)
        if vals is None:
            continue
        ar, sr, rm, rl, a1, a2, a3 = vals
        note = "  <- compare to SPIdepth" if key == "img_ms" else ""
        print("   {}  {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}  {:.4f}  {:.4f}{}".format(
            label, ar, sr, rm, rl, a1, a2, a3, note))


def _write_csv(path, suite_avg):
    header = ["variant"] + METRIC7
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for variant in VARIANTS:
            vals = suite_avg.get(variant)
            if vals is None:
                f.write(variant + "," + ",".join([""] * 7) + "\n")
            else:
                f.write(variant + "," + ",".join("{:.6f}".format(float(v))
                                                 for v in vals) + "\n")
    print("-> wrote {}".format(path))


def _convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if arg.strip():
            yield str(arg)


if __name__ == "__main__":
    options = MonodepthOptions()
    p = options.parser
    p.convert_arg_line_to_args = _convert_arg_line_to_args
    # Eval-specific arguments (not part of the inference parser).
    p.add_argument("--gt_depth_path", type=str, default="",
                   help="root containing the GT depth subdir")
    p.add_argument("--gt_depth_subdir", type=str,
                   default="data_depth_annotated/train/groundtruth_float32png",
                   help="path under gt_depth_path to the GT depth PNGs")
    p.add_argument("--gt_depth_encoding", type=str, default="float32_rgba",
                   help="auto | uint16 | float32_rgba")
    p.add_argument("--gt_depth_scale", type=float, default=100000.0,
                   help="uint16 PNG -> metres divisor")
    p.add_argument("--mask_subdir", type=str, default="masks",
                   help="per-folder subdir holding mask_XXXX.png stone masks")
    p.add_argument("--eval_split_file", type=str, default="",
                   help="split file to evaluate (default splits/<split>/val_files.txt)")
    p.add_argument("--eval_out", type=str, default="depth_eval_final.csv",
                   help="output CSV path for the final metric table")

    if len(sys.argv) == 2:
        opt = p.parse_args(["@" + sys.argv[1]])
    else:
        opt = p.parse_args()
    evaluate(opt)
