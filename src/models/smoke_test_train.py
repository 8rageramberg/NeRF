"""Smoke test entrypoint to validate environment/device wiring."""
from __future__ import annotations

from pathlib import Path

from .train_nerf import TrainConfig, train


def main() -> None:
    smoke_cfg = TrainConfig(
        num_views=6,
        azimuth_range=60.0,
        n_rays_per_image=64,
        n_pts_per_ray=32,
        n_iter=10,
        batch_size=2,
        lr=5e-4,
        log_every=2,
        output_dir=Path("outputs/smoke_test"),
        rotating_frames=6,
        rotating_rows=2,
        rotating_cols=3,
        eval_views=2,
        save_checkpoint=False,
    )
    train(smoke_cfg)


if __name__ == "__main__":
    main()
