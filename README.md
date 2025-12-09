# MoReMouse: Monocular Reconstruction of Laboratory Mouse

A PyTorch implementation of **MoReMouse** - a deep learning framework for dense 3D reconstruction of C57BL/6 laboratory mice from monocular images.

> Reference: [MoReMouse: Monocular Reconstruction of Laboratory Mouse](https://arxiv.org/abs/2507.04258)

## Overview

MoReMouse addresses the challenging problem of reconstructing detailed 3D surfaces of laboratory mice from single-view images. Key challenges include:
- Complex non-rigid deformations
- Textureless fur-covered surfaces
- Lack of realistic 3D mesh models

### Key Contributions

1. **Gaussian Mouse Avatar (AGAM)**: High-fidelity synthetic data generation via UV-based Gaussian splatting
2. **Triplane-based Reconstruction**: Transformer-based feedforward network with triplane representation
3. **Geodesic Embeddings**: Surface correspondence learning for semantic consistency
4. **Two-stage Training**: NeRF (volumetric) → DMTet (surface) for optimal reconstruction

## Architecture

```
Input Image → DINOv2 Encoder → Triplane Generator → NeRF/DMTet Renderer → Novel Views + 3D Mesh
                                     ↓
                              Multi-head MLP
                                     ↓
                        RGB + Density + Embedding
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
├── configs/
│   ├── config.yaml          # Main configuration
│   ├── model/               # Model configs
│   ├── data/                # Dataset configs
│   └── train/               # Training configs
├── src/
│   ├── models/
│   │   ├── mouse_body.py    # MAMMAL-based body model
│   │   ├── gaussian_avatar.py  # Gaussian avatar for data generation
│   │   ├── geodesic_embedding.py  # Surface correspondence
│   │   ├── triplane.py      # Triplane representation
│   │   └── moremouse_net.py # Main network
│   ├── data/
│   │   ├── dataset.py       # Dataset classes
│   │   └── transforms.py    # Data augmentation
│   ├── losses/              # Loss functions
│   └── utils/               # Utilities
├── scripts/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── inference.py         # Inference script
│   └── generate_synthetic_data.py  # Data generation
├── environment.yml
└── README.md
```

## Usage

### 1. Generate Synthetic Training Data

```bash
# Generate from MAMMAL mouse model
python scripts/generate_synthetic_data.py \
    --mouse-model /path/to/MAMMAL_mouse/mouse_model \
    --output data/synthetic \
    --num-frames 1000 \
    --num-views 64 \
    --device cuda:1
```

### 2. Training

```bash
# Train with default config
python scripts/train.py

# Train with custom settings
python scripts/train.py \
    experiment.name=my_experiment \
    train.stages.nerf.epochs=30 \
    logging.use_wandb=true

# Training on gpu05 (use GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/train.py experiment.device=cuda:0
```

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
- 140 articulated joints
- 13,059 vertices
- Linear Blend Skinning (LBS) deformation
- Euler angle rotation (ZYX convention)

### Gaussian Mouse Avatar
- UV-based Gaussian control
- Pose-dependent deformation
- Per-vertex Gaussian parameters (position, color, opacity, scale, rotation)

### Geodesic Embedding
- Heat method for geodesic distance computation
- 3D embedding preserving surface distances
- PCA → HSV transformation for visualization

### MoReMouse Network
- **Encoder**: DINOv2-B/14 (frozen, 768-dim features)
- **Triplane Generator**: 12-layer transformer, 64×64 resolution, 512 channels
- **Decoder**: 10-layer shared MLP with multi-head outputs
- **Renderer**: NeRF (128 samples/ray) or DMTet (256³ resolution)

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

- **Training**: NVIDIA GPU with 16GB+ VRAM (A100 recommended)
- **Inference**: NVIDIA GPU with 8GB+ VRAM
- **Memory**: 32GB RAM recommended
- **Storage**: 50GB+ for full dataset

## GPU05 Server Notes

```bash
# Use GPU 1 (GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=1

# Activate environment
conda activate moremouse

# Run training
python scripts/train.py experiment.device=cuda:0
```

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

- [MAMMAL](https://github.com/MAMMAL-Lab/MAMMAL) for the mouse body model
- [DINOv2](https://github.com/facebookresearch/dinov2) for image encoding
- [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian splatting
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) for DMTet implementation
