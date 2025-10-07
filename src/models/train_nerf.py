"""Training entrypoint for the simplified NeRF model from Chapter 6."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    EmissionAbsorptionRaymarcher,
    FoVPerspectiveCameras,
    ImplicitRenderer,
    MonteCarloRaysampler,
    NDCMultinomialRaysampler,
)

from utils.generate_cow_renders import generate_cow_renders
from utils.helper_functions import (
    generate_rotating_nerf,
    huber,
    sample_images_at_mc_locs,
    show_full_render,
)
from utils.plot_image_grid import image_grid
from .nerf_model import NeuralRadianceField


@dataclass
class TrainConfig:
    num_views: int = 40
    azimuth_range: float = 180.0
    n_rays_per_image: int = 750
    n_pts_per_ray: int = 128
    n_iter: int = 3000
    batch_size: int = 6
    lr: float = 1e-3
    lr_decay_at: float = 0.75
    lr_decay_factor: float = 0.1
    seed: int = 1
    log_every: int = 100
    output_dir: Path = Path("outputs")
    rotating_frames: int = 15
    rotating_rows: int = 3
    rotating_cols: int = 5
    device: Optional[str] = None
    eval_views: Optional[int] = None
    save_checkpoint: bool = True
    checkpoint_name: Optional[str] = None


def get_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    figs = output_dir / "figs"
    logs = output_dir / "logs"
    ckpts = output_dir / "ckpts"
    for path in (figs, logs, ckpts):
        path.mkdir(parents=True, exist_ok=True)
    return {"figs": figs, "logs": logs, "ckpts": ckpts}


def default_checkpoint_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"nerf_{timestamp}.pth"


@torch.no_grad()
def evaluate_model(
    neural_radiance_field: NeuralRadianceField,
    renderer_grid: ImplicitRenderer,
    target_cameras: FoVPerspectiveCameras,
    target_images: torch.Tensor,
    eval_views: Optional[int] = None,
) -> dict[str, object]:
    device = target_images.device
    num_views = len(target_images)
    indices = range(num_views) if eval_views is None else range(min(eval_views, num_views))

    psnr_scores: list[float] = []

    for idx in indices:
        camera = FoVPerspectiveCameras(
            R=target_cameras.R[idx][None],
            T=target_cameras.T[idx][None],
            znear=target_cameras.znear[idx][None],
            zfar=target_cameras.zfar[idx][None],
            aspect_ratio=target_cameras.aspect_ratio[idx][None],
            fov=target_cameras.fov[idx][None],
            device=device,
        )

        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batched_forward,
        )
        rendered_rgb = rendered_image_silhouette[0, ..., :3].clamp(0.0, 1.0).permute(2, 0, 1)
        target_rgb = target_images[idx].clamp(0.0, 1.0).permute(2, 0, 1)

        if rendered_rgb.shape[1:] != target_rgb.shape[1:]:
            rendered_rgb = F.interpolate(
                rendered_rgb[None], size=target_rgb.shape[1:], mode="bilinear", align_corners=False
            )[0]

        mse = torch.mean((rendered_rgb - target_rgb) ** 2)
        psnr = (-10.0 * torch.log10(mse.clamp_min(1e-10))).item()
        psnr_scores.append(psnr)

    metrics = {
        "psnr_per_view": psnr_scores,
        "psnr_mean": sum(psnr_scores) / len(psnr_scores) if psnr_scores else None,
        "num_views_evaluated": len(psnr_scores),
    }

    return metrics


def train(cfg: TrainConfig) -> None:
    device = get_device(cfg.device)
    torch.manual_seed(cfg.seed)

    target_cameras, target_images, target_silhouettes = generate_cow_renders(
        num_views=cfg.num_views, azimuth_range=cfg.azimuth_range
    )
    print(f"Generated {len(target_images)} images/silhouettes/cameras.")

    render_size = target_images.shape[1] * 2
    volume_extent_world = 3.0

    raysampler_mc = MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=cfg.n_rays_per_image,
        n_pts_per_ray=cfg.n_pts_per_ray,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

    raysampler_grid = NDCMultinomialRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=cfg.n_pts_per_ray,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)

    neural_radiance_field = NeuralRadianceField()

    renderer_grid = renderer_grid.to(device)
    renderer_mc = renderer_mc.to(device)
    target_cameras = target_cameras.to(device)
    target_images = target_images.to(device)
    target_silhouettes = target_silhouettes.to(device)
    neural_radiance_field = neural_radiance_field.to(device)

    optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=cfg.lr)

    loss_history_color: list[float] = []
    loss_history_sil: list[float] = []

    out_dirs = prepare_output_dirs(cfg.output_dir)

    for iteration in range(cfg.n_iter):
        if iteration == round(cfg.n_iter * cfg.lr_decay_at):
            print("Decreasing LR 10-fold ...")
            optimizer = torch.optim.Adam(
                neural_radiance_field.parameters(), lr=cfg.lr * cfg.lr_decay_factor
            )

        optimizer.zero_grad()
        batch_idx = torch.randperm(len(target_cameras))[: cfg.batch_size]

        batch_cameras = FoVPerspectiveCameras(
            R=target_cameras.R[batch_idx],
            T=target_cameras.T[batch_idx],
            znear=target_cameras.znear[batch_idx],
            zfar=target_cameras.zfar[batch_idx],
            aspect_ratio=target_cameras.aspect_ratio[batch_idx],
            fov=target_cameras.fov[batch_idx],
            device=device,
        )

        rendered_images_silhouettes, sampled_rays = renderer_mc(
            cameras=batch_cameras, volumetric_function=neural_radiance_field
        )

        rendered_images, rendered_silhouettes = rendered_images_silhouettes.split([3, 1], dim=-1)

        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None], sampled_rays.xys
        )
        sil_err = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()

        colors_at_rays = sample_images_at_mc_locs(target_images[batch_idx], sampled_rays.xys)
        color_err = huber(rendered_images, colors_at_rays).abs().mean()

        loss = color_err + sil_err
        loss_history_color.append(float(color_err))
        loss_history_sil.append(float(sil_err))

        loss.backward()
        optimizer.step()

        if cfg.log_every and iteration % cfg.log_every == 0:
            print(
                f"Iter {iteration:04d} | loss={loss.item():.4f} | color={color_err.item():.4f} | silhouette={sil_err.item():.4f}"
            )
            show_idx = torch.randperm(len(target_cameras))[:1]
            fig = show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R=target_cameras.R[show_idx],
                    T=target_cameras.T[show_idx],
                    znear=target_cameras.znear[show_idx],
                    zfar=target_cameras.zfar[show_idx],
                    aspect_ratio=target_cameras.aspect_ratio[show_idx],
                    fov=target_cameras.fov[show_idx],
                    device=device,
                ),
                target_images[show_idx][0],
                target_silhouettes[show_idx][0],
                renderer_grid,
                loss_history_color,
                loss_history_sil,
            )
            fig_path = out_dirs["figs"] / f"intermediate_{iteration:04d}.png"
            fig.savefig(fig_path)
            print(f"Saved preview to {fig_path}")

    with torch.no_grad():
        rotating_nerf_frames = generate_rotating_nerf(
            neural_radiance_field,
            target_cameras,
            renderer_grid,
            n_frames=cfg.rotating_frames,
            device=device,
        )

    grid_path = out_dirs["figs"] / "rotating_grid.png"
    fig = image_grid(
        rotating_nerf_frames.clamp(0.0, 1.0).cpu().numpy(),
        rows=cfg.rotating_rows,
        cols=cfg.rotating_cols,
        rgb=True,
        fill=True,
    )
    fig.savefig(grid_path)
    print(f"Saved rotating grid preview to {grid_path}")

    metrics = evaluate_model(
        neural_radiance_field,
        renderer_grid,
        target_cameras,
        target_images,
        cfg.eval_views,
    )

    metrics_path = out_dirs["logs"] / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if cfg.save_checkpoint:
        checkpoint_filename = cfg.checkpoint_name or default_checkpoint_name()
        checkpoint_path = out_dirs["ckpts"] / checkpoint_filename
        checkpoint = {
            "config": asdict(cfg),
            "state_dict": neural_radiance_field.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-views", type=int, default=TrainConfig.num_views)
    parser.add_argument("--azimuth-range", type=float, default=TrainConfig.azimuth_range)
    parser.add_argument("--n-rays-per-image", type=int, default=TrainConfig.n_rays_per_image)
    parser.add_argument("--n-pts-per-ray", type=int, default=TrainConfig.n_pts_per_ray)
    parser.add_argument("--n-iter", type=int, default=TrainConfig.n_iter)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--lr-decay-at", type=float, default=TrainConfig.lr_decay_at)
    parser.add_argument("--lr-decay-factor", type=float, default=TrainConfig.lr_decay_factor)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--rotating-frames", type=int, default=TrainConfig.rotating_frames)
    parser.add_argument("--rotating-rows", type=int, default=TrainConfig.rotating_rows)
    parser.add_argument("--rotating-cols", type=int, default=TrainConfig.rotating_cols)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--eval-views", type=int, default=TrainConfig.eval_views)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-name", type=str, default=TrainConfig.checkpoint_name)
    args = parser.parse_args()
    cfg_dict = vars(args)
    cfg_dict["save_checkpoint"] = not cfg_dict.pop("no_save_checkpoint")
    return TrainConfig(**cfg_dict)


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
