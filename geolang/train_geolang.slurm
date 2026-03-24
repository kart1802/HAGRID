#!/bin/bash
#SBATCH --job-name=geolang_train
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/geolang_%j.log
#SBATCH --error=logs/geolang_%j.err

mkdir -p logs
cd /scratch/tstanley/HAGRID/geolang || exit 1

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
# Run GeoLang training
# =============================
echo "=== START TRAINING ==="
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}
export WORLD_SIZE=${SLURM_NTASKS}
srun --kill-on-bad-exit=1 python train_geolang.py --config config/OCID-VLG/geolang_multiple_r50.yaml

# =============================
# GPU check after training
# =============================
echo "=== GPU AFTER ==="
nvidia-smi

echo "End time: $(date)"