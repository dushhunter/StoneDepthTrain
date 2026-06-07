# StoneVolMain — Stone Depth Estimation

## Quick Start

# 1. Self-supervised training (with GT depth supervision + MLflow, masks enabled in config)
python train.py ./configs/train_args.txt
# optional: override tracking URI from CLI
# python train.py ./configs/train_args.txt --mlflow_tracking_uri file:./mlruns

# 2. Inference / testing
python test_simple_SQL_config.py ./configs/infer_args.txt

# 3. Fine-tuning (metric depth)
python finetune/train_ft_SQLdepth.py ./configs/model_cvnXt.txt ./configs/finetune_args.txt

# 4. EXR → lossless float32 PNG conversion
python convert_exr_to_lossless_float32_png.py \
    --input_dir stone_syn_dataset/data_depth_annotated/train/groundtruth \
    --output_dir stone_syn_dataset/data_depth_annotated/train/groundtruth_float32png \
    --recursive --verify
```
# 5. 
python3 test_simple_SQL_all_weights.py ./configs/infer_args.txt --output_root test_results

# 6.
python ckpt_to_pth.py configs/model_cvnXt.txt exps/stone_syn_finetune/checkpoints/stone_syn_finetune_18-Apr_07-38-nodebs1-tep10-lr1e-06-wd0.01-bba41b71-4dac-4851-85fd-2f019086d969_best.pt finetune_weight

## Project Structure

```
StoneVolMain/
├── train.py                    # Self-supervised training entry point
├── test_simple_SQL_config.py   # Inference entry point
├── convert_exr_to_lossless_float32_png.py  # EXR→PNG converter
├── SQLdepth.py                 # Core model wrapper
├── trainer.py                  # Training loop
├── options.py                  # CLI argument definitions
├── layers.py                   # Depth/pose layer utilities
├── utils.py                    # General utilities
├── kitti_utils.py              # KITTI dataset utilities
├── requirements.txt            # Python dependencies
│
├── configs/                    # All configuration files
│   ├── train_args.txt          # train.py arguments
│   ├── infer_args.txt          # Inference arguments
│   ├── model_cvnXt.txt         # ConvNeXt-Large model config
│   └── finetune_args.txt       # Fine-tuning arguments
│
├── datasets/                   # Dataset loaders
├── networks/                   # Neural network modules
│
├── finetune/                   # Fine-tuning pipeline
│   ├── train_ft_SQLdepth.py    # Fine-tune entry point
│   ├── dataloader.py           # Data loading
│   ├── loss.py                 # Loss functions (SILog, L2)
│   ├── model_io.py             # Model I/O
│   ├── utils.py                # Fine-tune utilities
│   └── file_lists/             # Train/eval split files
│
├── splits/stone/               # Dataset split definitions
├── stone_syn_dataset/          # Training data (12 stones × 120 frames)
├── stone_weights/              # Pre-trained model weights
├── test_results/               # Inference output
└── docs/                       # Reference documents
```

## Architecture

- **Backbone:** ConvNeXt-Large (timm Unet encoder)
- **Decoder:** Depth_Decoder_QueryTr (transformer)
- **Resolution:** 576 × 1024
- **Depth range:** 0.01 – 1.0 m
- **GT encoding:** Float32 RGBA PNG (lossless, bit-exact)

## MLflow Tracking

MLflow is now integrated into both training flows and is designed to be safe:
if MLflow is not installed or the tracking server is unavailable, training keeps running.

Install MLflow:

```bash
pip install mlflow
```

The following configs already include `--mlflow`:

- `configs/train_args.txt`
- `configs/finetune_args.txt`

Main training uses options from `options.py`:

- `--mlflow`
- `--mlflow_tracking_uri`
- `--mlflow_experiment_name`
- `--mlflow_run_name`
- `--mlflow_tags` (comma-separated, `k=v,k2=v2`)
- `--mlflow_log_models`

Fine-tuning (`finetune/train_ft_SQLdepth.py`) supports the same MLflow flags.

Example with local file-based tracking:

```bash
python train.py ./configs/train_args.txt --mlflow_tracking_uri file:./mlruns
python finetune/train_ft_SQLdepth.py ./configs/model_cvnXt.txt ./configs/finetune_args.txt --mlflow_tracking_uri file:./mlruns
```

Convenience launcher (wraps the original scripts):

```bash
python run_with_mlflow.py train ./configs/train_args.txt --tracking_uri file:./mlruns
python run_with_mlflow.py finetune ./configs/model_cvnXt.txt ./configs/finetune_args.txt --tracking_uri file:./mlruns
```

To view runs:

```bash
python -m mlflow ui --backend-store-uri file:/home/dilan/msc_research/StoneVolMain/mlruns --host 0.0.0.0 --port 5000
```
