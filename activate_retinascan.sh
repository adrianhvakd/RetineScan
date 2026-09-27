#!/bin/bash
# Activar entorno virtual de RetinaScan con soporte CUDA
source "$(dirname "$0")/.venv/bin/activate"
export LD_LIBRARY_PATH="$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/curand/lib:$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
echo "RetinaScan environment activated"
echo "Usage:"
echo "  python retinascan/scripts/predict_image.py <imagen>"
echo "  python retinascan/scripts/predict_camera.py"
echo "  python retinascan/scripts/train.py"
