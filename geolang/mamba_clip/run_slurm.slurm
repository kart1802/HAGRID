#!/bin/bash
#SBATCH --job-name=vmamba_infer
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=vmamba_infer.%j.out
#SBATCH --error=vmamba_infer.%j.err

module purge
module load 2024r1
module load slurm
module load miniconda3
module load cuda/12.9

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlg

##Edit these parts based on where the file is located on Delftblue 
export PYTHONPATH=/home/svandeberk/CROG/HAGRID/geolang/mamba_clip/kernels/selective_scan:$PYTHONPATH
export LD_LIBRARY_PATH=/home/svandeberk/.conda/envs/vlg/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

##Edit this path based on where mamba_clip is on your folder
cd /home/svandeberk/CROG/HAGRID/geolang/mamba_clip
srun python infer_single_image.py