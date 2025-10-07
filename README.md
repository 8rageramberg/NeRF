# Simplified NeRF (Chapter 6)

This repository contains the Chapter 6 simplified NeRF implementation from *3D Deep Learning with Python* ported into a standalone training script. The code generates synthetic cow renders with PyTorch3D, fits a neural radiance field, and exports intermediates for inspection.

## Requirements
- Python 3.10+
- CUDA-capable GPU (optional but recommended). CPU will work for the smoke test and very small runs.
- PyTorch 2.0+, torchvision, PyTorch3D, NumPy, Matplotlib, tqdm. See `requirements.txt`.

> **Heads-up**: The first run will download the `cow_mesh` assets into `data/cow_mesh/` via `wget`. If outbound network is blocked on your cluster, download the three files (`cow.obj`, `cow.mtl`, `cow_texture.png`) yourself and place them there.

## Environment Setup
1. Create and activate an environment (conda shown; `python -m venv` works too).
   ```bash
   conda create -n nerf-ch6 python=3.10 -y
   conda activate nerf-ch6
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Set `PYTHONPATH` when running scripts from the repo root so Python can see `src/`.
   ```bash
   export PYTHONPATH=$(pwd)/src
   ```

## Run Order (Local)
1. **Smoke test** – verifies that the environment, renderer, and device all work. Completes within a couple of minutes on CPU.
   ```bash
   PYTHONPATH=$PWD/src python -m models.smoke_test_train --device cpu
   ```
   Feel free to switch `--device` to `cuda:0` to confirm GPU visibility.
2. **Main training** – launches the full Chapter 6 experiment (defaults: 3k iters, 40 views).
   ```bash
   python -m models.train_nerf --device cuda:0 --output-dir outputs/ch6_baseline
   ```
   Useful overrides:
   - `--n-iter 1500` for shorter runs
   - `--num-views 20` to halve data generation time
   - `--log-every 200` to reduce intermediate figure dumps
3. Inspect generated artefacts under `outputs/`. Figures land in `outputs/<run>/figs/`, metrics in `outputs/<run>/logs/metrics.json`, and the trained weights are saved to `outputs/<run>/ckpts/`.

## Running on a Slurm Cluster
1. Copy the repository (or at minimum `src/`, `requirements.txt`, and `scripts/train_nerf.sbatch`) to your project space.
2. Create the same Python environment on the cluster and install the requirements.
3. Adapt `scripts/train_nerf.sbatch` to match your site’s modules (CUDA, Python/conda module names, partition, GPU type, walltime, etc.).
4. Submit the job from the repo root:
   ```bash
   sbatch scripts/train_nerf.sbatch
   ```
   The batch script performs a smoke test first and then launches the long training run, writing Slurm logs into `logs/` and NeRF artefacts under `outputs/run_<jobid>/`.

### Customising the Slurm Job
- Pass CLI overrides directly in the sbatch script or with `sbatch --export=ALL,EXTRA_ARGS="--n-iter 1500 --batch-size 8" scripts/train_nerf.sbatch` and append `$EXTRA_ARGS` to the `python -m models.train_nerf` line.
- If your cluster lacks outbound network access, stage the cow mesh under `data/cow_mesh/` before submitting.
- For multi-GPU setups, duplicate the training call per device or wrap it with a distributed launcher (e.g. `torchrun`)—the current code trains on a single device.

## Repository Layout
```
NeRF/
├─ README.md
├─ requirements.txt
├─ scripts/
│  └─ train_nerf.sbatch
├─ src/
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ nerf_model.py
│  │  ├─ smoke_test_train.py
│  │  └─ train_nerf.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ generate_cow_renders.py
│     ├─ helper_functions.py
│     └─ plot_image_grid.py
├─ data/
│  └─ cow_mesh/  # auto-populated on first run
└─ outputs/
   ├─ ckpts/
   ├─ figs/
   └─ logs/
```

## Next Steps
- Extend `train_nerf.py` with checkpoint saving/restoring in `outputs/ckpts/`.
- Add evaluation utilities (PSNR/SSIM) and dataset loaders for real captures.
- Integrate experiment tracking (e.g. TensorBoard, Weights & Biases) if you need richer metrics.
