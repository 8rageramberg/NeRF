#!/usr/bin/env bash
#SBATCH --job-name=nerf-ch6-hgx
#SBATCH --partition=hgx2q
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/brage/D1/project/NeRF
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euxo pipefail

mkdir -p logs outputs

module purge
module load slurm/21.08.8
module load cuda12.4/toolkit/12.4.1

# Activate the torchenv environment configured with PyTorch 2.1.2 + cu118 + PyTorch3D 0.7.4
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)" || true
fi
conda activate /home/brage/miniconda3/envs/torchenv

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=$(pwd)/src

nvidia-smi || true
python -c "import torch; print('Torch', torch.__version__, 'CUDA available', torch.cuda.is_available())"

# Quick environment smoke test before launching the full run
python -m models.smoke_test_train --device cuda:0

exec python -m models.train_nerf \
  --device cuda:0 \
  --output-dir "outputs/run_${SLURM_JOB_ID}" \
  --log-every 200 \
  --rotating-frames 30
