#!/bin/bash
# MoReMouse Installation Script
#
# Handles correct installation order for PyTorch-dependent packages
# (torch-scatter, kaolin require PyTorch to be installed first)
#
# Usage:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh

set -e

echo "========================================"
echo "MoReMouse Environment Installation"
echo "========================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

ENV_NAME="moremouse"
PYTORCH_VERSION="2.0.1"
CUDA_VERSION="11.8"
PYTHON_VERSION="3.10"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  PyTorch: ${PYTORCH_VERSION}"
echo "  CUDA: ${CUDA_VERSION}"
echo ""

echo -e "${GREEN}[1/5] Creating conda environment...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Remove and recreate? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    fi
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

echo -e "${GREEN}[2/5] Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo -e "${GREEN}[3/5] Installing PyTorch (CUDA ${CUDA_VERSION})...${NC}"
conda install pytorch=${PYTORCH_VERSION} torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y

python -c "import torch; print(f'PyTorch {torch.__version__} installed, CUDA available: {torch.cuda.is_available()}')"

echo -e "${GREEN}[4/5] Installing torch-scatter and kaolin (pre-built wheels)...${NC}"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html || {
    echo -e "${YELLOW}Warning: kaolin installation failed. DMTet stage will not work.${NC}"
}

echo -e "${GREEN}[5/5] Installing remaining packages...${NC}"

pip install "numpy<2.0" scipy matplotlib
pip install gsplat>=1.5.0 || echo -e "${YELLOW}Warning: gsplat failed. Try: export CUDA_HOME=/usr/local/cuda-11.8${NC}"
pip install hydra-core>=1.3.0 omegaconf>=2.3.0
pip install transformers>=4.35.0 timm>=0.9.0
pip install trimesh>=4.0.0 "pyglet<2" networkx>=3.0 potpourri3d>=0.0.8
pip install zarr>=2.13.0 h5py>=3.8.0 opencv-python>=4.7.0 Pillow>=9.5.0 imageio>=2.31.0 imageio-ffmpeg>=0.4.8
pip install torchmetrics>=0.11.0 lpips>=0.1.4
pip install wandb>=0.15.0 tensorboard>=2.13.0
pip install python-dotenv>=1.0.0 tqdm>=4.65.0 joblib>=1.2.0 einops>=0.7.0 rich>=13.0.0

echo ""
echo "========================================"
echo -e "${GREEN}Installation complete!${NC}"
echo "========================================"
echo ""
echo "Activate environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Verify installation:"
echo "  python -c \"import torch; import gsplat; print('OK')\""
