#!/bin/bash
# Installation script for Q-GAD conda environment
# Usage: bash scripts/setup_env.sh

set -e  # Exit on error

echo "=================================="
echo "Q-GAD Environment Setup"
echo "=================================="
echo ""

# Detect CUDA
echo "Detecting CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "  CUDA detected: $CUDA_VERSION"

    # Check if we should use CUDA version
    read -p "  Use CUDA environment? (y/n): " use_cuda
    if [ "$use_cuda" = "y" ]; then
        ENV_FILE="environment-cuda.yml"
        echo "  Will use CUDA environment"
    else
        ENV_FILE="environment.yml"
        echo "  Will use CPU environment"
    fi
else
    echo "  No CUDA detected, using CPU environment"
    ENV_FILE="environment.yml"
fi

echo ""
echo "Step 1: Creating conda environment..."
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "  Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
conda env create -f "$ENV_FILE" || {
    echo "Environment creation failed, trying update..."
    conda env update -f "$ENV_FILE" --prune
}

echo ""
echo "Step 2: Activating environment..."
echo "=================================="

# Determine shell and source conda
if [ -n "$ZSH_VERSION" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Activate environment (use qgad or qgad-cuda)
if [ "$ENV_FILE" = "environment-cuda.yml" ]; then
    ENV_NAME="qgad-cuda"
else
    ENV_NAME="qgad"
fi

conda activate "$ENV_NAME"
echo "✓ Activated environment: $ENV_NAME"

echo ""
echo "Step 3: Verifying installations..."
echo "=================================="

# Check Python
python_version=$(python --version)
echo "  Python: $python_version"

# Check PyTorch
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Check CUDA (if applicable)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  CUDA: Available ($(python -c 'import torch; print(torch.version.cuda)'))"
else
    echo "  CUDA: Not available (using CPU)"
fi

# Check key packages
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "  NetworkX: $(python -c 'import networkx; print(networkx.__version__)')"

# Check DeepQuantum
echo ""
echo "Step 4: Checking DeepQuantum..."
echo "=================================="

if python -c "import deepquantum" 2>/dev/null; then
    echo "  ✓ DeepQuantum: Installed ($(python -c 'import deepquantum; print(getattr(deepquantum, \"__version__\", \"unknown\"))'))"

    # Try to test basic functionality
    echo "  Testing DeepQuantum..."
    python -c "
import deepquantum as dq
import numpy as np

# Test circuit creation
try:
    if hasattr(dq, 'QumodeCircuit'):
        cir = dq.QumodeCircuit(n_mode=4, backend='gaussian')
        print('  ✓ QumodeCircuit works')
    elif hasattr(dq, 'PhotonicCircuit'):
        cir = dq.PhotonicCircuit(n=4)
        print('  ✓ PhotonicCircuit works')
    elif hasattr(dq, 'Circuit'):
        cir = dq.Circuit(4)
        print('  ✓ Circuit works')
    else:
        print('  ? Circuit class unknown')
except Exception as e:
    print(f'  ✗ Circuit creation failed: {e}')
"
else
    echo "  ✗ DeepQuantum: Not installed or import failed"
    echo "    This is OK - the system will use mock implementation"
    echo "    To install: pip install git+https://github.com/turingq/deepquantum.git"
fi

echo ""
echo "Step 5: Setting up project structure..."
echo "=================================="

# Create necessary directories
mkdir -p data/{raw,processed,cache}
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints

echo "  ✓ Created project directories"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "  python scripts/verify_install.py"
echo ""
echo "To train a model with synthetic data:"
echo "  python main.py --mode train --preset testing --use_synthetic"
echo ""
