# MoReMouse: Monocular Reconstruction of Laboratory Mouse

[![Paper](https://img.shields.io/badge/arXiv-2507.04258-red)](https://arxiv.org/abs/2507.04258)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](docs/reports/251210_implementation_complete.md)

A PyTorch implementation of **MoReMouse** - a deep learning framework for dense 3D reconstruction of C57BL/6 laboratory mice from monocular images.

> Reference: [MoReMouse: Monocular Reconstruction of Laboratory Mouse](https://arxiv.org/abs/2507.04258)

## Overview

MoReMouse addresses the challenging problem of reconstructing detailed 3D surfaces of laboratory mice from single-view images. Key challenges include:
- Complex non-rigid deformations
- Textureless fur-covered surfaces
- Lack of realistic 3D mesh models

### Key Features

| Component | Description |
|-----------|-------------|
| **Gaussian Avatar (AGAM)** | UV-based Gaussian splatting for synthetic data generation |
| **Triplane Generator** | 12-layer transformer with Flash Attention |
| **NeRF Renderer** | 128 samples/ray volumetric rendering |
| **Two-stage Training** | NeRF (60 epochs) → DMTet (100 epochs) |
| **Paper-compliant** | All Table A3 specifications implemented |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MoReMouse Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Image     DINOv2        Triplane          NeRF         Output    │
│  [378×378]       Encoder       Generator         Decoder                │
│                                                                          │
│  ┌─────────┐    ┌─────────┐   ┌─────────────┐   ┌─────────┐   ┌───────┐│
│  │         │    │ ViT-B/14│   │ Transformer │   │ MLP     │   │ Novel ││
│  │   🐭    │ ─▶ │  768-d  │ ─▶│  12 layers  │ ─▶│ 10 layers│ ─▶│ Views ││
│  │         │    │  frozen │   │  128×128    │   │ RGB+σ+E │   │ + 3D  ││
│  └─────────┘    └─────────┘   └─────────────┘   └─────────┘   └───────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0.1 with CUDA 11.8
- NVIDIA GPU with 16GB+ VRAM (training), 8GB+ (inference)

### Quick Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/kafkapple/moremouse.git
cd moremouse

# Run installation script (handles correct package order)
chmod +x scripts/install.sh
./scripts/install.sh
```

### Manual Setup

If the installation script fails, install packages in this order:

```bash
# 1. Create conda environment
conda create -n moremouse python=3.10 -y
conda activate moremouse

# 2. Install PyTorch FIRST (required before torch-scatter/kaolin)
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install torch-scatter (pre-built wheel for PyTorch 2.0 + CUDA 11.8)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. Install kaolin (pre-built wheel, required for DMTet stage)
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html

# 5. Install remaining dependencies
pip install gsplat hydra-core omegaconf transformers timm
pip install trimesh "pyglet<2" networkx potpourri3d
pip install opencv-python lpips einops tqdm rich
```

### Troubleshooting

**torch-scatter build fails (`ModuleNotFoundError: torch`)**
- Cause: PyTorch not installed before torch-scatter
- Solution: Install from pre-built wheel:
  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
  ```

**gsplat CUDA compilation fails**
```bash
export CUDA_HOME=/usr/local/cuda-11.8
pip install gsplat --no-cache-dir
```

**NumPy version conflict**
```bash
pip install "numpy<2.0" --force-reinstall
```

**wandb build fails (missing Go binary)**
- Cause: Latest wandb tries to build from source
- Solution: Install specific version with pre-built wheel:
  ```bash
  pip install wandb==0.16.6 --only-binary :all:
  ```
- Note: wandb is optional - training works without it (use `logging.use_wandb=false`)

### Data Setup (MAMMAL Mouse Model)

The project uses relative paths by default. Place the MAMMAL_mouse data as a sibling directory:

```bash
# Expected directory structure
parent_dir/
├── moremouse/          # This project
└── MAMMAL_mouse/       # MAMMAL mouse data
    ├── data/
    │   └── markerless_mouse_1/
    └── mouse_model/
        ├── mouse.pkl
        └── mouse_txt/
```

**Option 1: Symlink (Recommended)**
```bash
# If MAMMAL_mouse is elsewhere, create a symlink
ln -s /path/to/MAMMAL_mouse ../MAMMAL_mouse
```

**Option 2: Environment Variables**
```bash
# Copy example env file
cp .env.example .env

# Set custom paths in .env
MAMMAL_DATA_PATH=/your/custom/path/MAMMAL_mouse/data
MOUSE_MODEL_PATH=/your/custom/path/MAMMAL_mouse/mouse_model
```

**Option 3: Command Line Override**
```bash
# Override at runtime
python scripts/generate_synthetic_data.py --mouse-model /path/to/mouse_model
```

## Project Structure

```
moremouse/
├── configs/                    # Hydra configuration system
│   ├── config.yaml             # Main config (paths, experiment)
│   ├── model/moremouse.yaml    # Model architecture (Paper Table A3)
│   ├── data/                   # Dataset configs
│   ├── train/default.yaml      # Training hyperparameters
│   └── avatar/default.yaml     # Avatar training config
├── src/
│   ├── models/
│   │   ├── mouse_body.py       # MAMMAL body (140 joints, 13059 vertices)
│   │   ├── gaussian_avatar.py  # AGAM + Trainer (400K iter)
│   │   ├── geodesic_embedding.py  # Heat method geodesic
│   │   ├── triplane.py         # Transformer + Decoder + Upsampler
│   │   └── moremouse_net.py    # DINOv2 + Triplane + NeRF
│   ├── data/
│   │   ├── dataset.py          # SyntheticDataset, RealDataset
│   │   ├── mammal_loader.py    # Multi-view loader (video/image)
│   │   └── transforms.py       # Data augmentation
│   ├── losses/
│   │   ├── reconstruction.py   # MSE, L1, SSIM, LPIPS
│   │   ├── mask.py, depth.py   # Mask/Depth losses
│   │   ├── geodesic.py         # Embedding loss
│   │   └── combined.py         # MoReMouseLoss (weighted sum)
│   └── utils/
│       ├── logging.py          # Console/file logging
│       ├── metrics.py          # PSNR, SSIM, LPIPS
│       └── visualization.py    # Multi-view grid, mesh rendering
├── scripts/
│   ├── train.py                # Two-stage training (Hydra)
│   ├── inference.py            # Single image/video inference
│   ├── evaluate.py             # Metrics evaluation
│   ├── generate_synthetic_data.py  # Avatar-based generation
│   ├── run_pipeline.py         # End-to-end pipeline
│   └── install.sh              # Dependency installation
├── docs/
│   ├── reports/                # Implementation reports
│   ├── guides/                 # Usage guides
│   └── pipeline_overview.md    # Technical documentation
├── environment.yml
└── README.md
```

## Usage

### Full Pipeline (run_pipeline.py)

```bash
# All stages: avatar → synthetic → train → evaluate → visualize
python scripts/run_pipeline.py --stage all \
    --data-dir ../MAMMAL_mouse/data/markerless_mouse_1 \
    --device cuda:0

# Individual stages
python scripts/run_pipeline.py --stage avatar    # AGAM training
python scripts/run_pipeline.py --stage synthetic # Data generation
python scripts/run_pipeline.py --stage train     # MoReMouse training
python scripts/run_pipeline.py --stage evaluate  # Metrics evaluation
python scripts/run_pipeline.py --stage visualize # Visualization
```

### 1. Generate Synthetic Training Data

```bash
# Quick test (50 frames, 8 views)
python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --mouse-model ../MAMMAL_mouse/mouse_model \
    --num-frames 50 \
    --num-views 8 \
    --device cuda:0

# Paper settings (6000 frames, 64 views)
python scripts/generate_synthetic_data.py \
    --output data/synthetic_full \
    --mouse-model ../MAMMAL_mouse/mouse_model \
    --num-frames 6000 \
    --num-views 64 \
    --image-size 800 \
    --device cuda:0
```

### 2. Training

```bash
# Quick test (5 epochs, no DMTet)
python scripts/train.py \
    experiment.name=quick_test \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=5 \
    train.stages.dmtet.epochs=0 \
    logging.use_wandb=false

# Paper settings (60 + 100 epochs)
python scripts/train.py \
    experiment.name=moremouse_full \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=60 \
    train.stages.dmtet.epochs=100

# Training on gpu05 (use GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/train.py experiment.device=cuda:0
```

**Key Hydra Config Overrides:**
| Config Key | Default | Description |
|------------|---------|-------------|
| `experiment.name` | `moremouse_default` | Experiment name |
| `experiment.device` | `cuda:0` | Device |
| `train.stages.nerf.epochs` | 60 | NeRF stage epochs |
| `train.stages.dmtet.epochs` | 100 | DMTet stage epochs |
| `data.dataloader.batch_size` | 4 | Batch size |
| `logging.use_wandb` | false | Enable W&B |

### 3. Evaluation

```bash
# Evaluate on synthetic data
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset synthetic

# Evaluate on real data
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset real
```

### 4. Inference

```bash
# Single image
python scripts/inference.py \
    --image path/to/mouse_image.png \
    --checkpoint checkpoints/best.pt \
    --num-views 8

# Video
python scripts/inference.py \
    --video path/to/mouse_video.mp4 \
    --checkpoint checkpoints/best.pt
```

## Model Components

### Mouse Body Model (MAMMAL-based)
- **140 articulated joints** with hierarchical structure
- **13,059 vertices** for detailed mesh representation
- **Linear Blend Skinning (LBS)** with adaptive bone length scaling
- **Axis-angle** rotation convention (MAMMAL format, 140 joints x 3)
- **22 semantic keypoints** extraction support

### Gaussian Mouse Avatar (AGAM)
- **UV-based Gaussian control**: 1 Gaussian per vertex (13,059 total)
- **Per-vertex parameters**: position offset, color (RGB), opacity, scale, rotation (quaternion)
- **400K iteration** training on 6-view multi-view data (800 frames)
- **Auto-checkpoint detection** for resume training
- **gsplat rendering** for differentiable Gaussian splatting

### Geodesic Embedding
- **Heat method** for geodesic distance computation (potpourri3d)
- **Dijkstra fallback** if Heat method unavailable
- **3D embedding** preserving surface distances
- **PCA → HSV** transformation for visualization

### MoReMouse Network (Paper Table A3 Compliant)
| Component | Specification |
|-----------|---------------|
| **Encoder** | DINOv2-B/14 (frozen, 768-dim) |
| **Triplane** | 12 layers, 16 heads, 128×128, 80 output channels |
| **Decoder** | 10 shared layers × 64 neurons |
| **NeRF** | 128 samples/ray, radius=0.87, near=0.1, far=4.0 |
| **Flash Attention** | O(n) memory via `F.scaled_dot_product_attention` |

## Training Details

### Two-Stage Training
1. **NeRF Stage** (60 epochs): Volumetric rendering for smooth gradient propagation
2. **DMTet Stage** (100 epochs): Explicit surface extraction for geometric detail

### Loss Functions
- **MSE Loss**: Pixel-wise RGB reconstruction (λ = 1.0)
- **LPIPS Loss**: Perceptual similarity (λ = 1.0)
- **Mask Loss**: Binary cross-entropy for opacity (λ = 0.3)
- **Smooth L1**: Large RGB discrepancy penalty (λ = 0.2)
- **Depth Loss**: Scale-invariant depth consistency (λ = 0.2)
- **Geodesic Loss**: Embedding consistency (λ = 0.1)

### Optimizer
- AdamW with lr=1e-5, weight_decay=0.01
- Cosine annealing LR scheduler
- 3000 warmup steps

## Metrics

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|---------|
| Synthetic | 22.03 | 0.966 | 0.053 |
| Real | 18.42 | - | - |

## Hardware Requirements

| Component | Training | Inference |
|-----------|----------|-----------|
| **GPU VRAM** | 16GB+ (A100 recommended) | 8GB+ |
| **RAM** | 32GB+ | 16GB+ |
| **Storage** | 100GB+ (full dataset) | 10GB |

## Quick Reference

```bash
# Environment setup
conda activate moremouse
export CUDA_VISIBLE_DEVICES=1

# Full pipeline (quick test)
python scripts/run_pipeline.py --stage synthetic --num-frames 50 --device cuda:0
python scripts/run_pipeline.py --stage train --nerf-epochs 5 --device cuda:0
python scripts/run_pipeline.py --stage evaluate --checkpoint checkpoints/latest.pt
python scripts/run_pipeline.py --stage visualize --image input.png

# Individual commands
python scripts/generate_synthetic_data.py --num-frames 50 --device cuda:0
python scripts/train.py experiment.device=cuda:0 train.stages.nerf.epochs=5
python scripts/evaluate.py --checkpoint checkpoints/best.pt --device cuda:0
python scripts/inference.py --image input.png --checkpoint checkpoints/best.pt
```

## Documentation

- [Implementation Status Report](docs/reports/251210_implementation_complete.md)
- [Pipeline Overview](docs/pipeline_overview.md)
- [GPU05 Pipeline Guide](docs/guides/gpu05_pipeline_guide.md)
- [Troubleshooting](docs/troubleshooting/)

## Data Source

- **Dataset**: markerless_mouse_1 (Dunn et al., 2021)
- **Camera System**: 6-view synchronized cameras
- **Avatar Training**: 800 frames (uniformly sampled from first 8000 frames)
- **Species**: C57BL/6 laboratory mice
- **Note**: "C57BL/6 mice usually show similar texture" (Paper)

## Citation

```bibtex
@article{moremouse2025,
  title={MoReMouse: Monocular Reconstruction of Laboratory Mouse},
  author={Zhong, Yuan and Sun, Jingxiang and Zhang, Zhongbin and An, Liang and Liu, Yebin},
  journal={arXiv preprint arXiv:2507.04258},
  year={2025}
}
```

## License

This project is for research purposes only.

## Acknowledgments

- [MAMMAL](https://github.com/MAMMAL-Lab/MAMMAL) for the mouse body model (140 joints, 13059 vertices)
- [DINOv2](https://github.com/facebookresearch/dinov2) for image encoding (ViT-B/14)
- [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian splatting
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) for DMTet implementation
- [potpourri3d](https://github.com/nmwsharp/potpourri3d) for geodesic distance (Heat method)
