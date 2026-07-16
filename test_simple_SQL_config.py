# StoneVol_main vs SPIdepth-main: this file differs (kept for comparison).
# python3 ./test_simple_SQL_config.py ./conf/cvnXt.txt
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
# from layers import disp_to_depth
from SQLdepth import MonodepthOptions, SQLdepth
STEREO_SCALE_FACTOR = 5.4


# --------------------------------------------------------------------------- #
# Edge-preserving guided-filter upsampling (pure NumPy, no extra dependency).
# Non-destructive: the output is a local affine (a*I + b) of the depth map, so
# the absolute metric scale is preserved (the .npy stays positive metres). It
# only snaps the soft 1/2-res depth boundary to the full-res image edges.
# --------------------------------------------------------------------------- #
def _box_sum(src, r):
    """He et al. O(1) box filter: sum over a (2r+1)x(2r+1) window, border-safe."""
    h, w = src.shape
    dst = np.zeros_like(src)

    cum = np.cumsum(src, axis=0)
    dst[0:r + 1, :] = cum[r:2 * r + 1, :]
    dst[r + 1:h - r, :] = cum[2 * r + 1:h, :] - cum[0:h - 2 * r - 1, :]
    dst[h - r:h, :] = np.tile(cum[h - 1:h, :], (r, 1)) - cum[h - 2 * r - 1:h - r - 1, :]

    cum = np.cumsum(dst, axis=1)
    dst[:, 0:r + 1] = cum[:, r:2 * r + 1]
    dst[:, r + 1:w - r] = cum[:, 2 * r + 1:w] - cum[:, 0:w - 2 * r - 1]
    dst[:, w - r:w] = np.tile(cum[:, w - 1:w], (1, r)) - cum[:, w - 2 * r - 1:w - r - 1]
    return dst


def _guided_filter(guide, src, r, eps):
    """Classic guided filter (He et al. 2010). guide/src are float32 2D, same shape."""
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    r = int(max(1, min(r, (min(guide.shape) - 1) // 2)))

    n = _box_sum(np.ones_like(guide), r)
    mean_I = _box_sum(guide, r) / n
    mean_p = _box_sum(src, r) / n
    mean_Ip = _box_sum(guide * src, r) / n
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _box_sum(guide * guide, r) / n
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_sum(a, r) / n
    mean_b = _box_sum(b, r) / n
    return (mean_a * guide + mean_b).astype(np.float32)


def _edge_upsample_depth(depth_full, rgb_image, radius, eps):
    """Sharpen a full-res depth map by snapping its edges to the RGB image edges."""
    guide = np.asarray(rgb_image.convert("L"), dtype=np.float32) / 255.0
    if guide.shape != depth_full.shape:
        guide = np.asarray(
            rgb_image.convert("L").resize(
                (depth_full.shape[1], depth_full.shape[0]), pil.LANCZOS),
            dtype=np.float32) / 255.0
    return _guided_filter(guide, depth_full.astype(np.float32), radius, eps)


# --------------------------------------------------------------------------- #
# Optional GT-based scale recovery (opt-in via --gt_scale, synthetic eval only).
# NON-destructive: applies a single positive global scalar. If GT is missing or
# unusable, it falls back to the RAW absolute depth (the old, independent
# behavior). It never fits/subtracts a plane and never produces relief.
# --------------------------------------------------------------------------- #
def _decode_float32_rgba_depth(path):
    """Decode a float32-in-RGBA-PNG depth map to a (H, W) float32 array of metres."""
    im = pil.open(path).convert("RGBA")
    arr = np.asarray(im, dtype=np.uint8)
    return arr.view(np.float32).reshape(arr.shape[0], arr.shape[1])


def _load_gt_depth(args, output_name):
    """Find and decode a GT depth map for the given frame name, or return None."""
    gt_dir = getattr(args, "gt_depth_dir", "") or ""
    if not gt_dir or not os.path.isdir(gt_dir):
        return None
    candidates = ["{}.png".format(output_name), "depth_{}.png".format(output_name)]
    try:
        candidates.append("depth_{:04d}.png".format(int(output_name)))
    except (ValueError, TypeError):
        pass
    encoding = getattr(args, "gt_depth_encoding", "float32_rgba")
    for name in candidates:
        path = os.path.join(gt_dir, name)
        if not os.path.isfile(path):
            continue
        if encoding == "float32_rgba":
            return _decode_float32_rgba_depth(path)
        elif encoding == "uint16":
            d = np.asarray(pil.open(path), dtype=np.float32)
            return d / float(getattr(args, "gt_depth_scale", 100000.0))
        else:  # auto
            im = pil.open(path)
            if im.mode in ("RGBA", "RGB"):
                return _decode_float32_rgba_depth(path)
            return np.asarray(im, dtype=np.float32) / float(
                getattr(args, "gt_depth_scale", 100000.0))
    return None


def _apply_gt_scale(args, depth_np, output_name):
    """Return depth aligned to the GT median if possible, else the RAW depth."""
    gt = _load_gt_depth(args, output_name)
    if gt is None:
        print("   gt_scale: no GT found for '{}' -> RAW absolute depth (old behavior)"
              .format(output_name))
        return depth_np
    if gt.shape != depth_np.shape:
        gt = np.asarray(
            pil.fromarray(gt).resize((depth_np.shape[1], depth_np.shape[0]), pil.NEAREST),
            dtype=np.float32)
    valid = (np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
             & (depth_np > 1e-6))
    if valid.sum() < 100:
        print("   gt_scale: too few valid GT pixels -> RAW absolute depth (old behavior)")
        return depth_np
    s = float(np.median(gt[valid]) / np.median(depth_np[valid]))
    print("   gt_scale: aligned to GT median, scale={:.4f}".format(s))
    return depth_np * s


def _apply_plane_anchor(args, depth_np):
    """Real-photo absolute anchor: rescale so the flat background sits at
    --ref_plane_depth metres. Positive global scalar only; never subtracts a
    plane (no relief)."""
    ref = float(getattr(args, "ref_plane_depth", -1.0))
    if ref <= 0:
        return depth_np
    bg = depth_np
    mask_dir = getattr(args, "stone_mask_path", "") or ""
    valid = np.isfinite(bg) & (bg > 1e-6)
    med = float(np.median(bg[valid])) if valid.any() else 0.0
    if med <= 1e-6:
        print("   plane_anchor: could not estimate background depth -> RAW depth")
        return depth_np
    s = ref / med
    print("   plane_anchor: background median -> {:.3f}m, scale={:.4f}".format(ref, s))
    return depth_np * s


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # assert args.model_name is not None, \
    #     "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ###CURSOR >>>
    #model = SQLdepth(opt)
    model = SQLdepth(args)
    ###<<<CURSOR

    feed_height = args.height
    feed_width = args.width
    model.to(device)
    model.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    edge_upsample = bool(getattr(args, "edge_upsample", False))
    eu_radius = int(getattr(args, "edge_upsample_radius", 8))
    eu_eps = float(getattr(args, "edge_upsample_eps", 1e-3))
    gt_scale = bool(getattr(args, "gt_scale", False))
    ref_plane_depth = float(getattr(args, "ref_plane_depth", -1.0))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image_orig = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image_orig.size
            input_image = input_image_orig.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            print(input_image.shape)
            # if True:
            if args.model_type == "nyu_pth_model":
                std_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                input_image = std_norm(input_image)


            # PREDICTION
            input_image = input_image.to(device)
            if args.model_type == "zoedepth":
                # outputs = model(input_image)["metric_depth"]
                outputs = model.infer(input_image)
            else:
                outputs = model(input_image)
                input_image_flipped = torch.flip(input_image, dims=[-1])
                outputs_flipped = model(input_image_flipped)
                outputs = (outputs + torch.flip(outputs_flipped, dims=[-1])) / 2.0

            disp = outputs
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            print(disp.shape, disp_resized.shape)
            output_name = os.path.splitext(os.path.basename(image_path))[0]

            # Full-resolution metric depth (raw network output = old behavior).
            disp_resized_np = disp_resized.squeeze().cpu().numpy().astype(np.float32)

            # (1) Edge-preserving guided upsampling (opt-in, keeps absolute scale).
            if edge_upsample:
                disp_resized_np = _edge_upsample_depth(
                    disp_resized_np, input_image_orig, eu_radius, eu_eps)

            # (2) Optional NON-destructive scale recovery. Default = raw depth.
            if gt_scale:
                disp_resized_np = _apply_gt_scale(args, disp_resized_np, output_name)
            elif ref_plane_depth > 0:
                disp_resized_np = _apply_plane_anchor(args, disp_resized_np)

            vmax = np.percentile(disp_resized_np, 95)

            # Saving uint16 depth map
            to_save_dir = os.path.join(output_directory, "uint16")
            if not os.path.exists(to_save_dir):
                os.makedirs(to_save_dir)
            to_save_path = os.path.join(to_save_dir, "{}.png".format(output_name))
            to_save = (disp_resized_np * 1000).astype('uint16')
            pil.fromarray(to_save).save(to_save_path)

            # Saving float32 NPY depth map (metric metres, for downstream 3D reconstruction)
            npy_save_dir = os.path.join(output_directory, "npy")
            if not os.path.exists(npy_save_dir):
                os.makedirs(npy_save_dir)
            npy_save_path = os.path.join(npy_save_dir, "{}.npy".format(output_name))
            np.save(npy_save_path, disp_resized_np.astype(np.float32))

            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r')
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}.jpeg".format(output_name))
            # plt.imsave(name_dest_im, disp_resized_np, cmap='gray') # for saving as gray depth maps
            im.save(name_dest_im) # for saving as colored depth maps

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(npy_save_path))

    print('-> Done!')


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == '__main__':
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()
    test_simple(opt)
