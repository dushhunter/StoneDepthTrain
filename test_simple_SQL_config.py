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


def _load_bg_mask(mask_dir, name, height, width):
    """Load a background mask (1 = background/flat surface, 0 = stone) or None.

    Stone masks follow the training convention (foreground stone > 127). Files may
    be named mask_<name>.<ext> or <name>.<ext>.
    """
    if not mask_dir:
        return None
    for cand in ("mask_{}.png".format(name), "{}.png".format(name),
                 "mask_{}.jpg".format(name), "{}.jpg".format(name)):
        p = os.path.join(mask_dir, cand)
        if os.path.isfile(p):
            m = pil.open(p).convert("L").resize((width, height), pil.NEAREST)
            return (np.array(m) <= 127).astype(np.float32)  # background = not stone
    return None


def _fit_background_plane(depth, bg_mask=None, iters=5, k=2.5, min_pixels=200):
    """Robustly fit a plane z = a*u + b*v + c to the flat background of a depth map.

    The flat surface dominates the image, so an iteratively reweighted least-squares
    fit (dropping >k MAD residuals each round) converges to the background plane and
    rejects the small stone bump even when no mask is supplied. Returns the full-frame
    plane depth [H, W] (float32) or None if the fit is not possible.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs / max(w - 1, 1)).astype(np.float64).ravel()
    v = (ys / max(h - 1, 1)).astype(np.float64).ravel()
    z = depth.astype(np.float64).ravel()
    valid = np.isfinite(z) & (z > 1e-6)
    if bg_mask is not None:
        valid &= (bg_mask.ravel() > 0.5)
    idx = np.where(valid)[0]
    if idx.size < min_pixels:
        return None
    A = np.stack([u, v, np.ones_like(u)], axis=1)
    sel = idx
    coeff = None
    for _ in range(iters):
        coeff, *_ = np.linalg.lstsq(A[sel], z[sel], rcond=None)
        resid = z[idx] - A[idx] @ coeff
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-9
        inl = np.abs(resid - med) < k * 1.4826 * mad
        new_sel = idx[inl]
        if new_sel.size < min_pixels or new_sel.size == sel.size:
            break
        sel = new_sel
    if coeff is None:
        return None
    return (A @ coeff).reshape(h, w).astype(np.float32)

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

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
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
            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            # --- Anchor absolute scale to the background plane -------------------
            # The network cannot infer an unseen stone's absolute distance from one
            # image, so its raw depth is off by a per-scene global scale/offset. Fit
            # the flat surface and either (a) rescale so the plane matches the rig's
            # known camera-to-surface distance (absolute metric), or (b) subtract the
            # plane to output stone relief (up-to-plane, no reference needed).
            bg_mask = _load_bg_mask(
                args.stone_mask_path, output_name, original_height, original_width)
            plane = _fit_background_plane(disp_resized_np, bg_mask)
            if plane is not None:
                cy, cx = original_height // 2, original_width // 2
                if args.ref_plane_depth > 0:
                    denom = float(plane[cy, cx])
                    if abs(denom) < 1e-6:
                        denom = float(np.median(plane))
                    scale = args.ref_plane_depth / denom
                    disp_resized_np = (disp_resized_np * scale).astype(np.float32)
                    print("   anchored (absolute): plane_center={:.4f}m -> ref={:.4f}m "
                          "(scale={:.4f})".format(denom, args.ref_plane_depth, scale))
                else:
                    # Height above the plane; +ve = protruding toward the camera.
                    disp_resized_np = (plane - disp_resized_np).astype(np.float32)
                    print("   anchored (relief): output is plane-relative height in m")
            else:
                print("   anchoring skipped (background plane fit failed)")
            # ---------------------------------------------------------------------

            vmax = np.percentile(disp_resized_np, 95)

            # Saving uint16 depth map
            to_save_dir = os.path.join(output_directory, "uint16")
            if not os.path.exists(to_save_dir):
                os.makedirs(to_save_dir)
            to_save_path = os.path.join(to_save_dir, "{}.png".format(output_name))
            # Clip negatives so relief-mode output (plane - depth) does not wrap uint16;
            # the float32 .npy below keeps the true signed values.
            to_save = (np.clip(disp_resized_np, 0, None) * 1000).astype('uint16')
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
            # print("   - {}".format(name_dest_npy))

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
