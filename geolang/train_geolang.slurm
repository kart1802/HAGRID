#!/bin/bash
#SBATCH --job-name=crog_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/crog_%j.log
#SBATCH --error=logs/crog_%j.err

# =============================
# Safety + debug info
# =============================
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

# =============================
# Activate conda environment
# =============================
source /home/tstanley/miniconda3/etc/profile.d/conda.sh
conda activate vmamba_copy

# Verify python + torch
echo "Python path: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# =============================
# GPU check before training
# =============================
echo "=== GPU BEFORE ==="
nvidia-smi

# =============================
# Run CROG training (2 GPUs)
# =============================
echo "=== START TRAINING ==="
python -m torch.distributed.run \
    --nproc_per_node=2 \
    train_crog.py --config config/OCID-VLG/crog_multiple_r50.yaml

# =============================
# GPU check after training
# =============================
echo "=== GPU AFTER ==="
nvidia-smi

echo "End time: $(date)"