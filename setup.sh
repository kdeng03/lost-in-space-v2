#!/bin/bash

# Create a Conda environment
echo "Creating Conda environment..."
conda create -n sr python=3.12 -y

# Activate the Conda environment
echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

# Install required dependencies
echo "Installing dependencies..."
# uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.19.1/vllm # Minimun version for trl with cuda 12.x support
uv pip install vllm==0.29.0 --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

echo "Conda environment setup complete."
