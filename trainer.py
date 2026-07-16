# StoneVol_main vs SPIdepth-main: this file differs (kept for comparison).
# pyright: reportGeneralTypeIssues=warning
from __future__ import absolute_import, division, print_function

import math
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
# import wandb
# from datetime import datetime as dt
# import uuid
from collections import OrderedDict
from mlflow_tracking import MLflowTracker, parse_mlflow_tags

PROJECT = "SQLdepth"
experiment_name="Mono"

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        mlflow_tags = parse_mlflow_tags(getattr(self.opt, "mlflow_tags", ""))
        mlflow_tags.setdefault("pipeline", "train.py")
        mlflow_tags.setdefault("dataset", getattr(self.opt, "dataset", "unknown"))
        mlflow_tags.setdefault("model_name", getattr(self.opt, "model_name", "mdp"))
        self.mlflow = MLflowTracker(
            enabled=getattr(self.opt, "mlflow", False),
            tracking_uri=(getattr(self.opt, "mlflow_tracking_uri", "") or None),
            experiment_name=(getattr(self.opt, "mlflow_experiment_name", "StoneVolMain-train") or "StoneVolMain-train"),
            run_name=(getattr(self.opt, "mlflow_run_name", "") or getattr(self.opt, "model_name", "mdp")),
            tags=mlflow_tags,
        )

        # checking height and width are multiples of 32
        # assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales) # default=[0], we only perform single scale training
        self.num_input_frames = len(self.opt.frame_ids) # default=[0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames # default=2 

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0]) # default=True

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # self.models["encoder"] = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        # self.models["encoder"] = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        if self.opt.backbone in ["resnet", "resnet_lite"]:
            self.models["encoder"] = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        elif self.opt.backbone == "resnet18_lite":
            self.models["encoder"] = networks.LiteResnetEncoderDecoder(model_dim=self.opt.model_dim)
        elif self.opt.backbone == "eff_b5":
            self.models["encoder"] = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        else: 
            self.models["encoder"] = networks.Unet(pretrained=(not self.opt.load_pretrained_model), backbone=self.opt.backbone, in_channels=3, num_classes=self.opt.model_dim, decoder_channels=self.opt.dec_channels, decoder_norm=getattr(self.opt, "decoder_norm", "group"))

        if self.opt.load_pretrained_model:
            print("-> Loading pretrained encoder from ", self.opt.load_pt_folder)
            encoder_path = os.path.join(self.opt.load_pt_folder, "encoder.pth")
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["encoder"].state_dict()}
            self.models["encoder"].load_state_dict(filtered_dict_enc)

        self.models["encoder"] = self.models["encoder"].cuda()
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"]) 
        # self.models["encoder"].to(self.device)
        
        
        prob_temperature = getattr(self.opt, "prob_temperature", 1.0)
        if self.opt.backbone.endswith("_lite"):
            self.models["depth"] = networks.Lite_Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        else:
            self.models["depth"] = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth, prob_temperature=prob_temperature)

        if self.opt.load_pretrained_model:
            print("-> Loading pretrained depth decoder from ", self.opt.load_pt_folder)
            depth_decoder_path = os.path.join(self.opt.load_pt_folder, "depth.pth")
            loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)

        self.models["depth"] = self.models["depth"].cuda()
#        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        # self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())


        self.models["pose"] = networks.PoseCNN(
            self.num_input_frames if self.opt.pose_model_input == "all" else 2) # default=2
        if self.opt.pretrained_pose :
            print(f'loaded pose from {self.opt.pose_net_path}')
            pose_net_path = os.path.join(self.opt.pose_net_path, 'pose.pth')
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in torch.load(pose_net_path).items()])
            self.models["pose"].load_state_dict(state_dict)
            print("-> Loading pretrained depth decoder from ", self.opt.pose_net_path)
            depth_decoder_path = os.path.join(self.opt.pose_net_path, "depth.pth")
            loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)

        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        # self.models["pose"].to(self.device)
        self.models["pose"] = self.models["pose"].cuda()

        # RGB-guided full-resolution refinement head. The depth decoder predicts
        # a half-resolution soft-binned map that blurs stone silhouettes when
        # upsampled; this head fuses the coarse depth with sharp RGB features and
        # predicts a residual to recover crisp edges at full input resolution.
        if getattr(self.opt, "use_edge_refine", False):
            self.models["refine"] = networks.EdgeRefine(
                base_channels=getattr(self.opt, "edge_refine_channels", 32),
                min_val=self.opt.min_depth, max_val=self.opt.max_depth)
            self.models["refine"] = self.models["refine"].cuda()
            self.models["refine"] = torch.nn.DataParallel(self.models["refine"])
            self.parameters_to_train += list(self.models["refine"].parameters())

        #self.models["pose"] = torch.nn.DataParallel(self.models["pose"])
        if self.opt.diff_lr :
            print("using diff lr for depth-net and pose-net")
            self.pose_params = []
            self.pose_params += list(self.models["encoder"].parameters())
        else :
            self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())

        # if self.opt.predictive_mask:
        #     assert self.opt.disable_automasking, \
        #         "When using predictive_mask, please disable automasking with --disable_automasking"

        #     # Our implementation of the predictive masking baseline has the the same architecture
        #     # as our depth decoder. We predict a separate mask for each source frame.
        #     self.models["predictive_mask"] = networks.DepthDecoder(
        #         self.models["encoder"].num_ch_enc, self.opt.scales,
        #         num_output_channels=(len(self.opt.frame_ids) - 1))
        #     self.models["predictive_mask"].to(self.device)
        #     self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        weight_decay = getattr(self.opt, "weight_decay", 0.0)
        if self.opt.diff_lr :
            df_params = [{"params": self.pose_params, "lr": self.opt.learning_rate / 10},
                      {"params": self.parameters_to_train, "lr": self.opt.learning_rate}]
            self.model_optimizer = optim.AdamW(df_params, lr=self.opt.learning_rate, weight_decay=weight_decay)
        else : 
            self.model_optimizer = optim.AdamW(self.parameters_to_train, lr=self.opt.learning_rate, weight_decay=weight_decay) # default lr=1e-4
        if getattr(self.opt, "use_cosine_lr", False):
            # Smoothly anneal to ~1% of the base LR over the whole run so the model
            # settles into its optimum instead of wandering at a constant LR (which
            # made the held-out metric peak early then diverge).
            self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.model_optimizer, T_max=self.opt.num_epochs,
                eta_min=self.opt.learning_rate * 0.01)
        else:
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1) # default=15

        #if self.opt.load_weights_folder is not None:
        #     self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir) # default to ~/tmp/mdp/train
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset,
                 "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                 "mc_dataset": datasets.MCDataset,
                 "stone": datasets.StoneDataset}  # StoneVol_main: adds "stone" dataset support
        self.dataset = datasets_dict[self.opt.dataset] # default="kitti"

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.dataset == "stone":
            train_dataset = self.dataset(
            self.opt.intrinsics_file_path, self.opt.data_path, train_filenames,
                self.opt.height, self.opt.width, self.opt.frame_ids, 1,
                is_train=True, img_ext=img_ext, use_mask=self.opt.use_mask,
                use_gt_depth=self.opt.use_gt_depth,
                gt_depth_path=self.opt.gt_depth_path if self.opt.gt_depth_path else self.opt.data_path,
                gt_depth_subdir=self.opt.gt_depth_subdir,
                gt_depth_encoding=self.opt.gt_depth_encoding,
                gt_depth_scale=self.opt.gt_depth_scale,
                use_strong_aug=getattr(self.opt, "use_strong_aug", False),
                use_crop_aug=getattr(self.opt, "use_crop_aug", False),
                use_scale_aug=getattr(self.opt, "use_scale_aug", False),
                scale_aug_max=getattr(self.opt, "scale_aug_max", 1.15),
                scale_aug_prob=getattr(self.opt, "scale_aug_prob", 0.5),
                cyclic_frames=False,
                frames_per_seq=getattr(self.opt, "frames_per_seq", 120))
        elif self.opt.dataset in ["mc_dataset"]:  # StoneVol_main: MonoDatasetMultiCam needs intrinsics_file_path
            train_dataset = self.dataset(
            self.opt.intrinsics_file_path, self.opt.data_path, train_filenames,  # StoneVol_main: pass intrinsics file first
                self.opt.height, self.opt.width, self.opt.frame_ids, 1,
                is_train=True, img_ext=img_ext)  # num_scales = 1
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 1, is_train=True, img_ext=img_ext) # num_scales = 1
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        if self.opt.dataset == "stone":
            val_dataset = self.dataset(
            self.opt.intrinsics_file_path, self.opt.data_path, val_filenames,
                self.opt.height, self.opt.width, self.opt.frame_ids, 1,
                is_train=False, img_ext=img_ext, use_mask=self.opt.use_mask,
                use_gt_depth=self.opt.use_gt_depth,
                gt_depth_path=self.opt.gt_depth_path if self.opt.gt_depth_path else self.opt.data_path,
                gt_depth_subdir=self.opt.gt_depth_subdir,
                gt_depth_encoding=self.opt.gt_depth_encoding,
                gt_depth_scale=self.opt.gt_depth_scale,
                cyclic_frames=False,
                frames_per_seq=getattr(self.opt, "frames_per_seq", 120))
        elif self.opt.dataset in ["mc_dataset"]:  # StoneVol_main: MonoDatasetMultiCam needs intrinsics_file_path
            val_dataset = self.dataset(
            self.opt.intrinsics_file_path, self.opt.data_path, val_filenames,  # StoneVol_main: pass intrinsics file first
                self.opt.height, self.opt.width, self.opt.frame_ids, 1,
                is_train=False, img_ext=img_ext)  # num_scales = 1
        else:
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 1, is_train=False, img_ext=img_ext) # num_scales = 1
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)

        self.save_opts()

    def _turntable_axis_depth(self, inputs):
        """Return the camera-to-turntable-axis distance (metres) used for the warp.

        If --turntable_axis_depth is positive, use it directly. Otherwise auto-estimate
        it from the median of the valid stone GT depth in this batch (GT is available
        during training); fall back to the middle of the depth range if no GT.
        """
        if getattr(self.opt, "turntable_axis_depth", -1.0) > 0:
            return float(self.opt.turntable_axis_depth)
        # Prefer the true per-folder metric axis distance if the dataset supplied it
        # (optional 6th column of the intrinsics file). This gives correct absolute
        # scale for the known-pose turntable warp without relying on GT at deployment.
        if "axis_depth" in inputs:
            try:
                return float(inputs["axis_depth"].float().mean().item())
            except Exception:
                pass
        if "depth_gt" in inputs:
            d = inputs["depth_gt"]
            valid = (d > self.opt.min_depth) & (d < self.opt.max_depth)
            if valid.any():
                return float(torch.median(d[valid]).item())
        return 0.5 * (self.opt.min_depth + self.opt.max_depth)

    def _turntable_transform(self, frame_id_offset, axis_depth):
        """Return a batched 4x4 rigid transform for a turntable offset of frame_id_offset frames.

        Camera is fixed; the object rotates by turntable_angle_deg per frame around the Y-axis.
        The rotation axis sits at the object, not the camera origin, so the correct
        camera-frame map is X' = R*X + (I - R)*c, where c is the axis position in camera
        coordinates: c = [axis_offset_x, 0, axis_depth]. Dropping the (I - R)*c translation
        (pure rotation about the camera origin) shifts neighbour frames by ~sin(angle)*depth
        and makes the photometric consistency loss inconsistent.
        """
        angle_rad = frame_id_offset * self.opt.turntable_angle_deg * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Y-axis rotation matrix (4x4 homogeneous)
        R = torch.tensor(
            [[ cos_a, 0.0, sin_a, 0.0],
             [   0.0, 1.0,   0.0, 0.0],
             [-sin_a, 0.0, cos_a, 0.0],
             [   0.0, 0.0,   0.0, 1.0]],
            dtype=torch.float32, device=self.device,
        )
        # Rotation centre (turntable axis) in camera coordinates.
        cx = float(getattr(self.opt, "turntable_axis_offset_x", 0.0))
        c = torch.tensor([cx, 0.0, float(axis_depth)], dtype=torch.float32, device=self.device)
        # t = (I - R3x3) @ c  -> rotate about the axis instead of the camera origin.
        R3 = R[:3, :3]
        t = (torch.eye(3, dtype=torch.float32, device=self.device) - R3) @ c
        T = R.clone()
        T[:3, 3] = t
        return T.unsqueeze(0).expand(self.opt.batch_size, -1, -1)  # [B, 4, 4]

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        run_status = "FINISHED"
        self.mlflow.start_run()
        self.mlflow.log_params(self.opt.__dict__, prefix="opt")
        self.mlflow.log_params(
            {
                "num_total_steps": self.num_total_steps,
                "num_scales": self.num_scales,
                "num_input_frames": self.num_input_frames,
                "num_pose_frames": self.num_pose_frames,
            },
            prefix="run",
        )

        try:
            self.epoch = 0
            self.step = 0
            # Best model is selected on the stone-region RMSE (metres) when a
            # foreground mask is available, else on the global scene RMSE.
            self.best_val_metric = float('inf')
            self.best_epoch = -1
            self.start_time = time.time()
            self.save_model()
            for self.epoch in range(self.opt.num_epochs):
                self.run_epoch()
                self.model_lr_scheduler.step()
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()
                val_metric = self.validate_full_epoch()
                if val_metric is not None and val_metric < self.best_val_metric:
                    self.best_val_metric = val_metric
                    self.best_epoch = self.epoch
                    self.save_model_best()
                    self.mlflow.log_metrics(
                        {"stone_rmse": self.best_val_metric, "stone_rmse_mm": self.best_val_metric * 1000,
                         "epoch": float(self.best_epoch)},
                        step=self.epoch,
                        prefix="best",
                    )
                    print("  ** Best model so far (stone_rmse={:.6f}m / {:.3f}mm) saved at epoch {}".format(
                        self.best_val_metric, self.best_val_metric * 1000, self.best_epoch))
            print("\n=== Training complete ===")
            print("  Best epoch: {}  (stone_rmse={:.6f}m / {:.3f}mm)".format(
                self.best_epoch, self.best_val_metric, self.best_val_metric * 1000))
            print("  Best weights: {}/models/weights_best/".format(self.log_path))

            self.mlflow.log_metrics(
                {
                    "best_stone_rmse": self.best_val_metric,
                    "best_stone_rmse_mm": self.best_val_metric * 1000,
                    "best_epoch": float(self.best_epoch),
                },
                step=self.epoch,
                prefix="final",
            )

            if getattr(self.opt, "mlflow_log_models", False):
                best_folder = os.path.join(self.log_path, "models", "weights_best")
                if os.path.isdir(best_folder):
                    self.mlflow.log_artifacts(best_folder, artifact_path="models/weights_best")
                latest_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
                if os.path.isdir(latest_folder):
                    self.mlflow.log_artifacts(latest_folder, artifact_path="models/final_weights")
                opt_path = os.path.join(self.log_path, "models", "opt.json")
                if os.path.isfile(opt_path):
                    self.mlflow.log_artifact(opt_path, artifact_path="models")
        except Exception:
            run_status = "FAILED"
            raise
        finally:
            self.mlflow.end_run(status=run_status)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        accum_steps = getattr(self.opt, 'accumulation_steps', 1)
        self.model_optimizer.zero_grad()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            scaled_loss = losses["loss"] / accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            duration = time.time() - before_op_time

            should_log = (batch_idx % self.opt.log_frequency == 0)

            if should_log:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.mlflow.log_metrics(
                    {"lr": self.model_optimizer.param_groups[0]["lr"]},
                    step=self.step,
                    prefix="train",
                )

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared": # default no
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        # Refine the coarse depth to full resolution using the input image as an
        # edge guide, so all downstream losses supervise the sharp map directly.
        if "refine" in self.models:
            outputs[("disp", 0)] = self.models["refine"](
                outputs[("disp", 0)], inputs[("color", 0, 0)])

        if self.opt.predictive_mask: # default no
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        # self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        if self.use_pose_net: # default=True
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.

        When --use_known_pose is active (turntable dataset), deterministic Y-axis
        rotation matrices are returned instead of running PoseCNN.
        """
        outputs = {}

        # Stone / turntable shortcut: bypass PoseCNN entirely and return known transforms.
        if self.opt.use_known_pose:
            axis_depth = self._turntable_axis_depth(inputs)
            dummy = torch.zeros(self.opt.batch_size, 1, 1, 3, device=self.device)
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = dummy
                    outputs[("translation", 0, f_i)] = dummy
                    outputs[("cam_T_cam", 0, f_i)] = self._turntable_transform(f_i, axis_depth)
            return outputs

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    # print(axisangle.shape)
                    # axisangle:[12, 1, 1, 3]  translation:[12, 1, 1, 3]
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    # outputs[("cam_T_cam", 0, f_i)]: [12, 4, 4]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next() # for old pytorch
            inputs = next(self.val_iter) # for new pytorch
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    @staticmethod
    def _fit_plane(depth_2d, bg_mask_2d, min_pixels=64):
        """Least-squares fit of a plane z = a*u + b*v + c to depth over a mask.

        u, v are pixel coordinates normalized to [0, 1]. Returns the fitted plane
        as a full [H, W] depth map, or None if there are too few background pixels.
        Used to measure the stone's height above its local background plane, which
        is invariant to the absolute depth scale and the plane's pose.
        """
        H, W = depth_2d.shape
        device = depth_2d.device
        m = bg_mask_2d.reshape(-1)
        if m.sum().item() < min_pixels:
            return None
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij")
        u = (xs / max(W - 1, 1)).reshape(-1)
        v = (ys / max(H - 1, 1)).reshape(-1)
        A = torch.stack([u, v, torch.ones_like(u)], dim=1)  # [H*W, 3]
        z = depth_2d.reshape(-1)
        A_bg = A[m]
        z_bg = z[m].unsqueeze(1)
        try:
            sol = torch.linalg.lstsq(A_bg, z_bg).solution  # [3, 1]
        except Exception:
            # Normal-equation fallback for older torch / singular systems.
            ata = A_bg.t() @ A_bg
            sol = torch.linalg.pinv(ata) @ (A_bg.t() @ z_bg)
        return (A @ sol).reshape(H, W)

    def _ms_scale(self, s_median):
        """Scale to use for the median-scaled ('ms') metric variant.

        Uses a fixed --pred_depth_scale_factor when the user set one (!=1), else the
        per-image median ratio gt/pred (the standard self-supervised monocular protocol).
        """
        pf = getattr(self.opt, "pred_depth_scale_factor", 1.0)
        if pf and pf != 1.0:
            return float(pf)
        return float(s_median)

    def _depth_metric_suite(self, pred_b, gt_b, region_mask, scale, min_pixels=64):
        """Return the 7 SPIdepth/KITTI metrics on a region, after applying `scale`.

        pred_b, gt_b: [H, W] depth maps. region_mask: [H, W] bool. Returns a length-7
        numpy array [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3] or None if too few
        pixels. Prediction is clamped to the configured depth range so log/ratio terms
        stay finite.
        """
        if region_mask.sum().item() < min_pixels:
            return None
        gt_sel = gt_b[region_mask]
        pred_sel = (pred_b[region_mask] * float(scale)).clamp(
            self.opt.min_depth, self.opt.max_depth)
        errs = compute_depth_errors(gt_sel, pred_sel)
        return np.array([float(e.item()) for e in errs], dtype=np.float64)

    def validate_full_epoch(self):
        """Run validation over the entire val set.

        Returns the selection metric (lower = better): the stone-region RMSE in
        metres when a foreground mask is available, otherwise the global RMSE.
        Returns None if GT depth is not available.

        Pixel-count-weighted accumulators are used (rather than averaging
        per-batch means) so the reported numbers are exact over the val set.
        Because the dominant flat surface drowns out the few-mm gem relief in a
        global metric, the stone-region RMSE (in mm) and mm-threshold accuracies
        are the metrics that actually reflect accuracy on the gemstone.
        """
        self.set_eval()
        # Global accumulators (whole valid scene).
        g_abs_rel_sum = 0.0
        g_se_sum = 0.0
        g_n = 0.0
        # Stone-region accumulators (foreground mask only).
        s_abs_rel_sum = 0.0
        s_se_sum = 0.0
        s_n = 0.0
        s_within = {1.0: 0.0, 2.0: 0.0, 5.0: 0.0}  # mm thresholds
        have_mask = False
        # Scale/shape decomposition accumulators (stone region, per-image):
        #   sms_*    : median-scaled stone SE (removes global depth scale)
        #   relief_* : plane-relative stone SE (removes scale AND plane pose)
        sms_se_sum = 0.0
        sms_n = 0.0
        relief_se_sum = 0.0
        relief_n = 0.0
        # SPIdepth-style 7-metric suite, averaged per image, for four variants:
        #   img_ms / img_abs     : whole valid image, median-scaled / absolute
        #   stone_ms / stone_abs : stone region,     median-scaled / absolute
        suite_variants = ("img_ms", "img_abs", "stone_ms", "stone_abs")
        suite_sum = {k: np.zeros(7, dtype=np.float64) for k in suite_variants}
        suite_cnt = {k: 0 for k in suite_variants}

        with torch.no_grad():
            for inputs in self.val_loader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)

                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["depth"](features)
                if "refine" in self.models:
                    outputs[("disp", 0)] = self.models["refine"](
                        outputs[("disp", 0)], inputs[("color", 0, 0)])

                if "depth_gt" not in inputs:
                    self.set_train()
                    return None

                depth_pred = outputs[("disp", 0)]
                depth_gt = inputs["depth_gt"]

                if depth_pred.dim() == 3:
                    depth_pred = depth_pred.unsqueeze(1)
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(1)

                if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                    depth_pred = F.interpolate(
                        depth_pred, depth_gt.shape[-2:], mode="bilinear", align_corners=False)

                depth_pred = torch.clamp(depth_pred, self.opt.min_depth, self.opt.max_depth)
                valid = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)

                if valid.sum() == 0:
                    continue

                gt_v = depth_gt[valid]
                pred_v = depth_pred[valid]
                err_v = pred_v - gt_v
                g_abs_rel_sum += (torch.abs(err_v) / gt_v).sum().item()
                g_se_sum += (err_v ** 2).sum().item()
                g_n += float(valid.sum().item())

                # SPIdepth-style 7-metric suite on the whole valid image (per image).
                for b in range(depth_gt.shape[0]):
                    gt_b = depth_gt[b, 0]
                    pred_b = depth_pred[b, 0]
                    valid_b = valid[b, 0]
                    if valid_b.sum().item() == 0:
                        continue
                    s_img = (torch.median(gt_b[valid_b]) /
                             torch.median(pred_b[valid_b]).clamp(min=1e-6)).item()
                    m_abs = self._depth_metric_suite(pred_b, gt_b, valid_b, 1.0)
                    if m_abs is not None:
                        suite_sum["img_abs"] += m_abs
                        suite_cnt["img_abs"] += 1
                    if not self.opt.disable_median_scaling:
                        m_ms = self._depth_metric_suite(
                            pred_b, gt_b, valid_b, self._ms_scale(s_img))
                        if m_ms is not None:
                            suite_sum["img_ms"] += m_ms
                            suite_cnt["img_ms"] += 1

                # Stone-region metrics restricted to the foreground mask.
                if ("mask", 0, 0) in inputs:
                    have_mask = True
                    fg = inputs[("mask", 0, 0)]
                    if fg.shape[-2:] != depth_gt.shape[-2:]:
                        fg = F.interpolate(fg, depth_gt.shape[-2:], mode="nearest")
                    stone_valid = valid & (fg > 0.5)
                    if stone_valid.sum() > 0:
                        gt_s = depth_gt[stone_valid]
                        pred_s = depth_pred[stone_valid]
                        err_s = pred_s - gt_s
                        abs_err_mm = torch.abs(err_s) * 1000.0
                        s_abs_rel_sum += (torch.abs(err_s) / gt_s).sum().item()
                        s_se_sum += (err_s ** 2).sum().item()
                        s_n += float(stone_valid.sum().item())
                        for thr in s_within:
                            s_within[thr] += float((abs_err_mm < thr).sum().item())

                    # Scale/shape decomposition, computed per image in the batch.
                    for b in range(depth_gt.shape[0]):
                        gt_b = depth_gt[b, 0]
                        pred_b = depth_pred[b, 0]
                        valid_b = valid[b, 0]
                        stone_b = stone_valid[b, 0]
                        if stone_b.sum().item() == 0 or valid_b.sum().item() == 0:
                            continue
                        gt_sb = gt_b[stone_b]

                        # (1) Median-scaled: align global scale, then stone SE.
                        pred_med = torch.median(pred_b[valid_b]).clamp(min=1e-6)
                        scale = torch.median(gt_b[valid_b]) / pred_med
                        pred_sb_scaled = pred_b[stone_b] * scale
                        sms_se_sum += ((pred_sb_scaled - gt_sb) ** 2).sum().item()
                        sms_n += float(stone_b.sum().item())

                        # SPIdepth-style 7-metric suite on the stone region.
                        m_sabs = self._depth_metric_suite(pred_b, gt_b, stone_b, 1.0)
                        if m_sabs is not None:
                            suite_sum["stone_abs"] += m_sabs
                            suite_cnt["stone_abs"] += 1
                        if not self.opt.disable_median_scaling:
                            m_sms = self._depth_metric_suite(
                                pred_b, gt_b, stone_b, self._ms_scale(scale.item()))
                            if m_sms is not None:
                                suite_sum["stone_ms"] += m_sms
                                suite_cnt["stone_ms"] += 1

                        # (2) Plane-relative: subtract each map's own background plane,
                        # then compare stone height-above-plane (scale+pose invariant).
                        bg_b = valid_b & (~stone_b)
                        plane_pred = self._fit_plane(pred_b, bg_b)
                        plane_gt = self._fit_plane(gt_b, bg_b)
                        if plane_pred is not None and plane_gt is not None:
                            relief_pred = pred_b[stone_b] - plane_pred[stone_b]
                            relief_gt = gt_b[stone_b] - plane_gt[stone_b]
                            relief_se_sum += ((relief_pred - relief_gt) ** 2).sum().item()
                            relief_n += float(stone_b.sum().item())

        self.set_train()

        if g_n == 0:
            return None

        g_abs_rel = g_abs_rel_sum / g_n
        g_rmse = (g_se_sum / g_n) ** 0.5
        mlflow_metrics = {}
        selection_metric = g_rmse  # fall back to scene RMSE if no mask

        # Stone-region / median-scaled values are computed only to drive model
        # selection; they are no longer reported (we report the 7 SPIdepth metrics).
        s_rmse_medscaled_mm = (sms_se_sum / sms_n) ** 0.5 * 1000 if sms_n > 0 else None

        if have_mask and s_n > 0:
            selection_metric = (s_se_sum / s_n) ** 0.5  # absolute stone-region RMSE

        # --select_on_medscaled selects on the median-scaled stone RMSE (anchored-at-
        # deploy protocol); otherwise the absolute stone RMSE above is used.
        if getattr(self.opt, "select_on_medscaled", False) and s_rmse_medscaled_mm is not None:
            selection_metric = s_rmse_medscaled_mm / 1000.0

        # SPIdepth 7-metric suite (per-image mean). We report only the whole-image,
        # median-scaled variant (img_ms) - the exact SPIdepth protocol.
        suite_avg = {k: (suite_sum[k] / suite_cnt[k]) if suite_cnt[k] > 0 else None
                     for k in suite_variants}

        metric7 = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        img_ms = suite_avg.get("img_ms")
        if img_ms is not None:
            print("  Epoch {} val (whole-image, median-scaled; SPIdepth protocol): "
                  "AbsRel={:.4f} SqRel={:.4f} RMSE={:.4f} RMSElog={:.4f} "
                  "d1={:.4f} d2={:.4f} d3={:.4f}".format(self.epoch, *img_ms))
            for name, v in zip(metric7, img_ms):
                self.writers["val"].add_scalar(
                    "epoch/{}".format(name), float(v), self.epoch)
                mlflow_metrics[name] = float(v)

        self.mlflow.log_metrics(mlflow_metrics, step=self.epoch, prefix="val_epoch")

        # Persist the 7 SPIdepth metrics (whole-image, median-scaled) per epoch.
        self._append_val_metrics_csv(suite_avg)

        return selection_metric

    def _append_val_metrics_csv(self, suite_avg):
        """Append the 7 SPIdepth metrics (whole-image, median-scaled) for this epoch.

        Written to {log_path}/val_metrics.csv as:
        epoch, AbsRel, SqRel, RMSE, RMSE(log), delta<1.25, delta<1.25^2, delta<1.25^3
        - the standard SPIdepth protocol (per-image median-aligned to GT). The row is
        left blank for epochs where the median-scaled variant was unavailable.
        """
        csv_path = os.path.join(self.log_path, "val_metrics.csv")
        header = ["epoch", "abs_rel", "sq_rel", "rmse", "rmse_log",
                  "delta_1.25", "delta_1.25^2", "delta_1.25^3"]

        vals = suite_avg.get("img_ms")
        if vals is None:
            row = [str(self.epoch)] + [""] * 7
        else:
            row = [str(self.epoch)] + ["{:.6f}".format(float(v)) for v in vals]

        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            f.write(",".join(row) + "\n")
        print("  -> val metrics appended to {}".format(csv_path))

    @staticmethod
    def _to_scalar(value):
        """Best-effort conversion to scalar float for external loggers."""
        try:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return float(value.detach().mean().cpu().item())
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                return float(np.mean(value))
            return float(value)
        except Exception:
            return None

    def save_model_best(self):
        """Save the best model weights to a fixed folder for easy access."""
        save_folder = os.path.join(self.log_path, "models", "weights_best")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if model_name == 'pose':
                to_save = model.state_dict()
            else:
                to_save = model.module.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        with open(os.path.join(save_folder, "best_epoch.txt"), "w") as f:
            f.write("epoch: {}\nstone_rmse_m: {:.6f}\nstone_rmse_mm: {:.3f}\n".format(
                self.best_epoch, self.best_val_metric, self.best_val_metric * 1000))

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                depth = disp
            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                # For posecnn the translation is scaled by mean inverse depth unless
                # known poses are already set (turntable mode bypasses this block).
                if self.opt.pose_model_type == "posecnn" and not self.opt.use_stereo \
                        and not self.opt.use_known_pose:

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                # pix_coords: [bs, h, w, 2]

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_smoothness_loss(self, disp, img):
        """Compute edge-aware depth smoothness loss.
        
        Depth should be smooth away from image edges. Computed on disparity (inverse depth).
        Args:
            disp: [B, 1, H, W] disparity map
            img: [B, 3, H, W] RGB image for edge detection
        Returns:
            scalar smoothness loss
        """
        # Keep both tensors at the same resolution before gradient computation.
        # Disparity resolution can differ from input color resolution depending on decoder setup.
        if img.shape[-2:] != disp.shape[-2:]:
            img = F.interpolate(img, size=disp.shape[-2:], mode="bilinear", align_corners=False)

        # Compute gradients of disparity
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        # Compute gradients of image (for edge detection)
        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Weight smoothness by inverse image gradients (edge-aware)
        weights_x = torch.exp(-grad_img_x)
        weights_y = torch.exp(-grad_img_y)
        
        smoothness_x = grad_disp_x * weights_x
        smoothness_y = grad_disp_y * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()

    def compute_multiview_consistency_loss(self, inputs, outputs):
        """Compute photometric consistency loss between adjacent frames.
        
        For turntable data, adjacent frames should have similar depth (photometric consistency).
        Warp adjacent frame into reference frame using predicted depth + known pose.
        
        Returns: scalar consistency loss (or 0 if not applicable)
        """
        # Only apply for stone dataset with use_known_pose + turntable structure
        if not getattr(self.opt, 'use_known_pose', False):
            return torch.tensor(0.0, device=self.device)
        
        if 1 not in self.opt.frame_ids or -1 not in self.opt.frame_ids:
            return torch.tensor(0.0, device=self.device)
        
        try:
            scale = 0  # Use single scale
            img_ref = inputs[("color", 0, scale)]      # [B, 3, H, W]

            # Per-pixel photometric reprojection error for each available neighbour.
            reproj = []
            if ("color", 1, scale) in outputs:
                reproj.append(
                    self.compute_reprojection_loss(outputs[("color", 1, scale)], img_ref))
            if ("color", -1, scale) in outputs:
                reproj.append(
                    self.compute_reprojection_loss(outputs[("color", -1, scale)], img_ref))

            if len(reproj) == 0:
                return torch.tensor(0.0, device=self.device)

            # Occlusion-robust: keep the best-matching neighbour per pixel (min, not mean).
            reproj = torch.cat(reproj, dim=1)          # [B, N, H, W]
            per_pixel, _ = reproj.min(dim=1, keepdim=True)  # [B, 1, H, W]

            # Restrict to the foreground stone: specular highlights and the (moving)
            # background violate brightness constancy and would corrupt the constraint.
            mask = None
            if self.opt.use_mask and (("mask", 0, scale) in inputs):
                mask = inputs[("mask", 0, scale)]
                if mask.shape[-2:] != per_pixel.shape[-2:]:
                    mask = F.interpolate(mask, size=per_pixel.shape[-2:], mode="nearest")
            if mask is not None:
                denom = mask.sum().clamp(min=1.0)
                return (per_pixel * mask).sum() / denom
            return per_pixel.mean()
        except Exception:
            # Silently fail if frames not available
            return torch.tensor(0.0, device=self.device)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # Self-supervised photometric reprojection + disparity smoothness loss.
        # When training with GT depth this is disabled by default so the model is
        # fully GT-supervised (the photometric term assumes Lambertian surfaces and
        # fights the absolute metric scale / specular gem highlights). Pass
        # --use_photometric to force it back on.
        use_photometric = getattr(self.opt, "use_photometric", False) or \
            not getattr(self.opt, "use_gt_depth", False)

        for scale in (self.opt.scales if use_photometric else []):
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            mask = None
            if self.opt.use_mask and (("mask", 0, source_scale) in inputs):
                mask = inputs[("mask", 0, source_scale)]
                # broadcast mask to all source frames
                reprojection_losses = reprojection_losses * mask

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if mask is not None:
                    identity_reprojection_losses = identity_reprojection_losses * mask

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            if color.shape[-2:] != disp.shape[-2:]:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # if GPU memory is not enough, you can downsample color instead
            # color = F.interpolate(color, [self.opt.height // 2, self.opt.width // 2], mode="bilinear", align_corners=False)
            smooth_loss = 0
            smooth_loss = get_smooth_loss(norm_disp, color)
            # smooth_loss
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # GT metric depth supervised loss (stone dataset with --use_gt_depth).
        # Applied once at full resolution (scale 0) on valid stone pixels.
        if "depth_gt" in inputs and getattr(self.opt, "use_gt_depth", False):
            depth_pred = outputs[("depth", 0, 0)]
            depth_gt = inputs["depth_gt"]  # [B, 1, H, W] float32 metres
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_pred_gt = F.interpolate(
                    depth_pred, depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            else:
                depth_pred_gt = depth_pred
            valid = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
            if valid.sum() > 0:
                # Per-pixel weight map that focuses supervision on the stone.
                # The flat surface dominates the image, so without weighting the
                # gem's few-mm relief contributes almost nothing to the loss.
                #   weight = 1 + (stone_w - 1) * mask   (up-weight the stone body)
                #          + boundary_w * silhouette_ring (up-weight the gem edge)
                weight = torch.ones_like(depth_gt)
                fg = None
                if getattr(self.opt, "use_mask", False) and (("mask", 0, 0) in inputs):
                    fg = inputs[("mask", 0, 0)].float()
                    if fg.shape[-2:] != depth_gt.shape[-2:]:
                        fg = F.interpolate(fg, depth_gt.shape[-2:], mode="nearest")
                    stone_w = getattr(self.opt, "stone_loss_weight", 1.0)
                    if stone_w != 1.0:
                        weight = weight + (stone_w - 1.0) * fg
                    boundary_w = getattr(self.opt, "boundary_loss_weight", 0.0)
                    if boundary_w > 0:
                        # Silhouette ring = dilate(mask) - erode(mask). Erosion is a
                        # min-pool, implemented as -maxpool(-mask).
                        dil = F.max_pool2d(fg, kernel_size=3, stride=1, padding=1)
                        ero = -F.max_pool2d(-fg, kernel_size=3, stride=1, padding=1)
                        ring = (dil - ero).clamp(0.0, 1.0)
                        weight = weight + boundary_w * ring

                # Restrict weights to valid pixels and form a weighted mean.
                w_valid = weight * valid.float()
                denom = w_valid.sum().clamp(min=1.0)
                abs_diff = torch.abs(depth_pred_gt - depth_gt)

                if getattr(self.opt, "use_berhu_loss", False):
                    # BerHu (reverse Huber): L1 for small errors, L2 for large ones.
                    c = 0.2 * abs_diff[valid].max().clamp(min=1e-7)
                    berhu = torch.where(abs_diff <= c, abs_diff,
                                        (abs_diff ** 2 + c ** 2) / (2.0 * c))
                    gt_primary = (w_valid * berhu).sum() / denom
                    losses["loss/gt_berhu"] = gt_primary
                else:
                    gt_primary = (w_valid * abs_diff).sum() / denom
                    losses["loss/gt_l1"] = gt_primary

                # Scale-invariant log term. Default weight is 0 so the objective is
                # absolute (mm-accurate); enable via --gt_silog_weight if desired.
                log_diff = torch.log(depth_pred_gt[valid]) - torch.log(depth_gt[valid])
                gt_log = torch.sqrt(
                    torch.var(log_diff) + 0.15 * torch.pow(torch.mean(log_diff), 2))
                silog_w = getattr(self.opt, "gt_silog_weight", 0.0)
                gt_loss = gt_primary + silog_w * gt_log
                losses["loss/gt_depth"] = gt_loss
                losses["loss/gt_silog"] = gt_log
                total_loss += self.opt.gt_depth_weight * gt_loss

                # Background-plane anchor: extra absolute L1 on the flat-surface
                # (non-stone) pixels. On a fixed camera/surface rig the surface depth
                # determines the absolute scale, so pinning it hard forces the model
                # to be metric on its own instead of only scale-relatively correct.
                plane_weight = getattr(self.opt, 'gt_plane_weight', 0.0)
                if plane_weight > 0 and fg is not None:
                    bg = valid & (fg < 0.5)
                    bg_n = bg.float().sum().clamp(min=1.0)
                    if bg.any():
                        plane_loss = (torch.abs(depth_pred_gt - depth_gt) * bg.float()).sum() / bg_n
                        losses["loss/gt_plane"] = plane_loss
                        total_loss += plane_weight * plane_loss

                # Depth-gradient loss: force the model to match surface slopes,
                # not just absolute depth values. This captures curvature/detail and,
                # crucially, the sharp depth cliff at the stone silhouette.
                #
                # The loss is weighted by the same per-pixel `weight` map used above
                # (1 + stone emphasis + boundary-ring emphasis). Because the depth
                # decoder predicts an expectation over bins, it tends to round off the
                # silhouette; heavily up-weighting the gradient error on the boundary
                # ring pushes the model to reproduce the steep edge instead of blurring
                # it. Uses a weighted mean so the emphasis is a true reweighting.
                grad_weight = getattr(self.opt, 'gt_grad_weight', 0.0)
                if grad_weight > 0:
                    pred_dx = depth_pred_gt[:, :, :, 1:] - depth_pred_gt[:, :, :, :-1]
                    pred_dy = depth_pred_gt[:, :, 1:, :] - depth_pred_gt[:, :, :-1, :]
                    gt_dx = depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1]
                    gt_dy = depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :]
                    valid_dx = (valid[:, :, :, 1:] & valid[:, :, :, :-1]).float()
                    valid_dy = (valid[:, :, 1:, :] & valid[:, :, :-1, :]).float()
                    # Edge-position weight = max of the two neighbouring pixel weights,
                    # so a difference straddling the silhouette gets the boundary boost.
                    w_dx = torch.maximum(weight[:, :, :, 1:], weight[:, :, :, :-1]) * valid_dx
                    w_dy = torch.maximum(weight[:, :, 1:, :], weight[:, :, :-1, :]) * valid_dy
                    grad_loss = torch.tensor(0.0, device=self.device)
                    if w_dx.sum() > 0:
                        grad_loss = grad_loss + (w_dx * torch.abs(pred_dx - gt_dx)).sum() / w_dx.sum().clamp(min=1.0)
                    if w_dy.sum() > 0:
                        grad_loss = grad_loss + (w_dy * torch.abs(pred_dy - gt_dy)).sum() / w_dy.sum().clamp(min=1.0)
                    losses["loss/gt_grad"] = grad_loss
                    total_loss += grad_weight * grad_loss

                # Surface-normal loss: penalise angular difference between
                # predicted and GT surface normals derived from depth.
                normal_weight = getattr(self.opt, 'gt_normal_weight', 0.0)
                if normal_weight > 0:
                    def _depth_to_normals(d):
                        dx = d[:, :, :, 1:] - d[:, :, :, :-1]
                        dy = d[:, :, 1:, :] - d[:, :, :-1, :]
                        dx = F.pad(dx, (0, 1, 0, 0))
                        dy = F.pad(dy, (0, 0, 0, 1))
                        ones = torch.ones_like(dx)
                        n = torch.cat([-dx, -dy, ones], dim=1)
                        return F.normalize(n, dim=1)
                    pred_n = _depth_to_normals(depth_pred_gt)
                    gt_n = _depth_to_normals(depth_gt)
                    cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
                    valid_n = valid.float()
                    if valid_n.sum() > 0:
                        normal_loss = (1.0 - cos_sim) * valid_n
                        normal_loss = normal_loss.sum() / valid_n.sum().clamp(min=1)
                        losses["loss/gt_normal"] = normal_loss
                        total_loss += normal_weight * normal_loss

                # Plane-relative relief loss: isolate the few-mm stone relief from the
                # dominant (and easy) background plane. Fit a plane to GT and to the
                # prediction over the background, subtract it from each, and match the
                # residual height over the stone. This is invariant to the absolute
                # scale and the plane's pose, so it supervises the stone's *shape*
                # directly, on top of the absolute BerHu term.
                relief_weight = getattr(self.opt, 'gt_relief_weight', 0.0)
                if relief_weight > 0 and fg is not None:
                    relief_terms = []
                    for b in range(depth_gt.shape[0]):
                        vb = valid[b, 0]
                        fgb = fg[b, 0] > 0.5
                        bg_b = vb & (~fgb)   # background = valid, non-stone
                        st_b = vb & fgb      # stone region
                        if st_b.sum() < 16:
                            continue
                        plane_gt = self._fit_plane(depth_gt[b, 0], bg_b)
                        plane_pr = self._fit_plane(depth_pred_gt[b, 0], bg_b)
                        if plane_gt is None or plane_pr is None:
                            continue
                        relief_gt = depth_gt[b, 0] - plane_gt
                        relief_pr = depth_pred_gt[b, 0] - plane_pr
                        relief_terms.append(torch.abs(relief_pr - relief_gt)[st_b].mean())
                    if relief_terms:
                        relief_loss = torch.stack(relief_terms).mean()
                        losses["loss/gt_relief"] = relief_loss
                        total_loss += relief_weight * relief_loss

                # Curvature (Laplacian) loss: match the second-order surface variation
                # so fine stone detail/facets are reproduced, not just first-order
                # slopes. Weighted by the same stone/boundary map over the valid interior.
                curv_weight = getattr(self.opt, 'gt_curvature_weight', 0.0)
                if curv_weight > 0:
                    def _laplacian(d):
                        return (d[:, :, 2:, 1:-1] + d[:, :, :-2, 1:-1]
                                + d[:, :, 1:-1, 2:] + d[:, :, 1:-1, :-2]
                                - 4.0 * d[:, :, 1:-1, 1:-1])
                    pred_lap = _laplacian(depth_pred_gt)
                    gt_lap = _laplacian(depth_gt)
                    w_in = (weight[:, :, 1:-1, 1:-1] * valid[:, :, 1:-1, 1:-1].float())
                    if w_in.sum() > 0:
                        curv_loss = (w_in * torch.abs(pred_lap - gt_lap)).sum() / w_in.sum().clamp(min=1.0)
                        losses["loss/gt_curvature"] = curv_loss
                        total_loss += curv_weight * curv_loss

        # ENHANCEMENT #2: Depth smoothness regularization (edge-aware)
        # Encourages smooth depth predictions away from image edges
        if getattr(self.opt, 'use_smoothness_loss', False):
            scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            smoothness = self.compute_smoothness_loss(disp, color)
            losses["loss/smoothness"] = smoothness
            smooth_weight = getattr(self.opt, 'smoothness_weight', 0.001)
            total_loss += smooth_weight * smoothness

        # ENHANCEMENT #1: Multi-view consistency loss (photometric)
        # For turntable data, enforces geometric consistency across views.
        # This runs on top of GT supervision (independent of --use_photometric): it is a
        # masked, metric-scale cross-view constraint, NOT the full monocular self-supervised
        # reprojection loop (which fights absolute scale / specular gem highlights).
        if getattr(self.opt, 'use_multiview_loss', False):
            consistency = self.compute_multiview_consistency_loss(inputs, outputs)
            if consistency > 0:
                losses["loss/consistency"] = consistency
                consistency_weight = getattr(self.opt, 'consistency_weight', 0.1)
                total_loss += consistency_weight * consistency

        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training.

        Stone dataset: evaluates on all valid GT pixels within the stone depth
        range, with no KITTI-style crop and no median-scaling of predictions.
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_gt = inputs["depth_gt"]

        # Normalize both tensors to [B, 1, H, W] for safe masking/indexing.
        while depth_gt.dim() > 4:
            depth_gt = depth_gt.squeeze(1)
        while depth_pred.dim() > 4:
            depth_pred = depth_pred.squeeze(1)
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)
        if depth_gt.shape[1] != 1:
            depth_gt = depth_gt[:, :1]
        if depth_pred.shape[1] != 1:
            depth_pred = depth_pred[:, :1]

        # Resize prediction to GT resolution if needed (no KITTI-specific size).
        if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
            depth_pred = F.interpolate(
                depth_pred, depth_gt.shape[-2:], mode="bilinear", align_corners=False)
        depth_pred = torch.clamp(depth_pred, self.opt.min_depth, self.opt.max_depth)
        depth_pred = depth_pred.detach()

        # Valid mask: GT must be positive and within the configured depth range.
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
        if mask.sum() == 0:
            return

        depth_gt = torch.masked_select(depth_gt, mask)
        depth_pred = torch.masked_select(depth_pred, mask)
        # No median scaling: model is trained with metric supervision.

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        mlflow_metrics = {}
        for l, v in losses.items():
            scalar = self._to_scalar(v)
            if scalar is not None and np.isfinite(scalar):
                mlflow_metrics[l] = scalar
        if mlflow_metrics:
            self.mlflow.log_metrics(mlflow_metrics, step=self.step, prefix=mode)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0 and (("color", frame_id, s) in outputs):
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking and \
                        ("identity_selection/{}".format(s) in outputs):
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk, default /home/Process3/tmp/mdp/models/
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            # for nn.DataParallel models, you must use model.module.state_dict() instead of model.state_dict()
            if model_name == 'pose':
               to_save = model.state_dict()
            else:
                to_save = model.module.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
