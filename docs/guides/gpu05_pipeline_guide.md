# MoReMouse GPU05 Pipeline Guide

**Date**: 2024-12-09
**Environment**: gpu05 (SSH)
**GPU**: CUDA_VISIBLE_DEVICES=1 (GPU 0 occupied)

---

## Directory Structure

```
/home/joon/
├── dev/
│   └── moremouse/              # Project root
│       ├── configs/            # Hydra configs
│       ├── scripts/            # Executable scripts
│       ├── src/                # Source code
│       ├── data/               # Generated data
│       │   └── synthetic/      # Synthetic training data
│       ├── checkpoints/        # Model checkpoints
│       ├── outputs/            # Training outputs
│       ├── results/            # Inference results
│       └── logs/               # Training logs
│
└── MAMMAL_mouse/               # External data (sibling dir or symlink)
    ├── data/
    │   └── markerless_mouse_1/ # Source video frames
    └── mouse_model/
        ├── mouse.pkl           # MAMMAL model (required)
        └── mouse_txt/          # Auxiliary files
```

---

## Step 0: Environment Setup

### 0.1 SSH Connection
```bash
ssh gpu05
cd ~/dev/moremouse
```

### 0.2 Installation (First time only)
```bash
# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### 0.3 Activate Environment
```bash
conda activate moremouse

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 0.4 GPU Selection
```bash
# gpu05: Use GPU 1 (GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=1

# Verify
nvidia-smi
```

### 0.5 Data Path Setup

**Option A: Symlink (Recommended)**
```bash
ln -s /path/to/MAMMAL_mouse ~/MAMMAL_mouse
```

**Option B: Environment Variables**
```bash
export MAMMAL_DATA_PATH=/path/to/MAMMAL_mouse/data
export MOUSE_MODEL_PATH=/path/to/MAMMAL_mouse/mouse_model
```

**Verify paths**:
```bash
ls ~/MAMMAL_mouse/mouse_model/mouse.pkl  # Should exist
```

---

## Step 1: Generate Synthetic Data

### 1.1 Command
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --mouse-model ../MAMMAL_mouse/mouse_model \
    --num-frames 100 \
    --num-views 8 \
    --image-size 378 \
    --device cuda:0
```

### 1.2 Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `data/synthetic` | Output directory |
| `--mouse-model` | Auto-detect | Path to mouse_model folder |
| `--num-frames` | 100 | Number of poses to render |
| `--num-views` | 8 | Viewpoints per pose |
| `--image-size` | 378 | Output image size (DINOv2 input) |
| `--device` | `cuda:1` | PyTorch device |

### 1.3 Output Structure
```
data/synthetic/
├── images/
│   ├── frame_0000_view_00.png
│   ├── frame_0000_view_01.png
│   └── ...
├── cameras/
│   ├── frame_0000_view_00.json  # {viewmat, K, position}
│   └── ...
└── metadata.json                 # Dataset info
```

### 1.4 Full Paper Settings (12K scenes)
```bash
# Warning: Takes several hours
CUDA_VISIBLE_DEVICES=1 python scripts/generate_synthetic_data.py \
    --output data/synthetic_full \
    --num-frames 6000 \
    --num-views 64 \
    --image-size 800 \
    --device cuda:0
```

---

## Step 2: Training

### 2.1 Quick Test (5 epochs)
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    experiment.name=test_run \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=5 \
    train.stages.dmtet.epochs=0 \
    data.dataloader.batch_size=2 \
    logging.use_wandb=false
```

### 2.2 Full Training (Paper settings)
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    experiment.name=moremouse_full \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=60 \
    train.stages.dmtet.epochs=100 \
    data.dataloader.batch_size=4 \
    logging.use_wandb=false
```

### 2.3 Training with W&B Logging
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    experiment.name=moremouse_wandb \
    experiment.device=cuda:0 \
    logging.use_wandb=true \
    logging.wandb_project=moremouse
```

### 2.4 Hydra Config Overrides
| Config Key | Default | Description |
|------------|---------|-------------|
| `experiment.name` | `moremouse_default` | Experiment name |
| `experiment.device` | `cuda:0` | Device (use with CUDA_VISIBLE_DEVICES) |
| `experiment.seed` | 42 | Random seed |
| `train.stages.nerf.epochs` | 60 | NeRF stage epochs |
| `train.stages.dmtet.epochs` | 100 | DMTet stage epochs |
| `train.optimizer.lr` | 1e-5 | Learning rate |
| `train.training.mixed_precision` | true | AMP (FP16) |
| `data.dataloader.batch_size` | 4 | Batch size |
| `logging.use_wandb` | false | Enable W&B |
| `logging.log_freq` | 100 | Log every N steps |

### 2.5 Output Structure
```
outputs/moremouse_full/YYYY-MM-DD_HH-MM-SS/
├── config.yaml           # Saved config
├── train.log             # Training log
└── tensorboard/          # TensorBoard logs

checkpoints/
├── latest.pt             # Latest checkpoint
├── best.pt               # Best PSNR checkpoint
└── checkpoint_XXXXXXXX.pt
```

### 2.6 Monitor Training
```bash
# TensorBoard
tensorboard --logdir outputs/ --port 6006

# Tail logs
tail -f outputs/*/train.log
```

---

## Step 3: Evaluation

### 3.1 Evaluate on Synthetic Data
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset synthetic \
    --data-dir data \
    --output results/eval_synthetic.json \
    --device cuda:0 \
    --batch-size 4
```

### 3.2 Evaluate on Real Data
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset real \
    --data-dir data \
    --output results/eval_real.json \
    --device cuda:0
```

### 3.3 Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Model checkpoint path |
| `--dataset` | `synthetic` | Dataset: synthetic/real/both |
| `--data-dir` | `data` | Data directory |
| `--output` | `results/evaluation.json` | Output JSON path |
| `--device` | `cuda:1` | Device |
| `--batch-size` | 4 | Evaluation batch size |

### 3.4 Metrics Output
```json
{
  "psnr": 22.03,
  "ssim": 0.966,
  "lpips": 0.053,
  "iou": 0.94,
  "num_samples": 1000
}
```

---

## Step 4: Inference & Visualization

### 4.1 Single Image Inference
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
    --image path/to/input.png \
    --checkpoint checkpoints/best.pt \
    --output results/inference \
    --num-views 8 \
    --device cuda:0
```

### 4.2 Video Inference
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
    --video path/to/input.mp4 \
    --checkpoint checkpoints/best.pt \
    --output results/video_inference \
    --num-views 4 \
    --device cuda:0
```

### 4.3 Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | - | Input image path |
| `--video` | - | Input video path |
| `--checkpoint` | Required | Model checkpoint |
| `--output` | `results/inference` | Output directory |
| `--num-views` | 8 | Number of novel views |
| `--device` | `cuda:1` | Device |

### 4.4 Output Structure
```
results/inference/
├── input.png             # Input image
├── view_00.png           # Novel view 0 (front)
├── view_01.png           # Novel view 1
├── view_02.png           # Novel view 2
├── ...
├── view_07.png           # Novel view 7
├── rotation.mp4          # 360° rotation video
└── depth.png             # Depth map (if available)
```

---

## Step 5: Full Pipeline Script

### 5.1 Quick Pipeline (Test)
```bash
#!/bin/bash
# quick_pipeline.sh

set -e
export CUDA_VISIBLE_DEVICES=1
cd ~/dev/moremouse
conda activate moremouse

echo "=== Step 1: Generate Data (Small) ==="
python scripts/generate_synthetic_data.py \
    --output data/synthetic_test \
    --num-frames 50 \
    --num-views 4 \
    --device cuda:0

echo "=== Step 2: Train (5 epochs) ==="
python scripts/train.py \
    experiment.name=quick_test \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=5 \
    train.stages.dmtet.epochs=0 \
    data.dataloader.batch_size=2

echo "=== Step 3: Evaluate ==="
python scripts/evaluate.py \
    --checkpoint checkpoints/latest.pt \
    --dataset synthetic \
    --device cuda:0

echo "=== Step 4: Inference ==="
python scripts/inference.py \
    --image data/synthetic_test/images/frame_0000_view_00.png \
    --checkpoint checkpoints/latest.pt \
    --output results/quick_test \
    --device cuda:0

echo "=== Done! Check results/quick_test/ ==="
```

### 5.2 Full Pipeline (Production)
```bash
#!/bin/bash
# full_pipeline.sh

set -e
export CUDA_VISIBLE_DEVICES=1
cd ~/dev/moremouse
conda activate moremouse

# Generate full dataset
python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --num-frames 1000 \
    --num-views 64 \
    --device cuda:0

# Train with paper settings
python scripts/train.py \
    experiment.name=moremouse_prod \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=60 \
    train.stages.dmtet.epochs=100

# Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset both \
    --device cuda:0

# Inference on test images
for img in data/test_images/*.png; do
    python scripts/inference.py \
        --image "$img" \
        --checkpoint checkpoints/best.pt \
        --output results/inference/$(basename "$img" .png) \
        --device cuda:0
done
```

---

## Troubleshooting

### CUDA OOM Error
```bash
# Reduce batch size
python scripts/train.py data.dataloader.batch_size=2

# Check GPU memory
nvidia-smi -l 1
```

### wandb Build Error
```bash
pip install wandb==0.16.6 --only-binary :all:
# Or disable wandb
python scripts/train.py logging.use_wandb=false
```

### Device Mismatch
```bash
# Always use cuda:0 with CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1
python scripts/train.py experiment.device=cuda:0
```

### Missing MAMMAL Data
```bash
# Check symlink
ls -la ~/MAMMAL_mouse

# Create if needed
ln -s /actual/path/to/MAMMAL_mouse ~/MAMMAL_mouse
```

---

## Quick Reference Card

```bash
# Environment
ssh gpu05 && cd ~/dev/moremouse && conda activate moremouse
export CUDA_VISIBLE_DEVICES=1

# Generate data
python scripts/generate_synthetic_data.py --num-frames 100 --device cuda:0

# Train
python scripts/train.py experiment.device=cuda:0 logging.use_wandb=false

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best.pt --device cuda:0

# Inference
python scripts/inference.py --image input.png --checkpoint checkpoints/best.pt --device cuda:0
```
