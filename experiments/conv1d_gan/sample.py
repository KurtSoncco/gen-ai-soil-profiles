from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from config import cfg
from data import create_dataloader
from models import Generator1D


def load_latest_checkpoint(dir_path: str) -> Optional[str]:
    files = [
        f
        for f in os.listdir(dir_path)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(dir_path, files[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument(
        "--truncate", type=str, default="none", choices=["none", "heuristic"]
    )  # heuristic truncates by empirical length distribution
    args = parser.parse_args()

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We need the max_length for generator shape
    _, max_length = create_dataloader(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )

    G = Generator1D(
        latent_dim=cfg.latent_dim, base_ch=cfg.base_channels, out_length=max_length
    ).to(device)
    ckpt_path = load_latest_checkpoint(cfg.out_dir)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoints found. Train the model first.")
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state["G"])  # type: ignore[arg-type]
    G.eval()

    z = torch.randn(args.num, cfg.latent_dim, device=device)
    with torch.no_grad():
        samples = G(z).cpu().numpy()  # (N, 1, L)

    if args.truncate == "heuristic":
        # Estimate empirical length distribution by mask lengths from dataloader
        loader, _ = create_dataloader(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
        )
        lengths = []
        for _, mask in loader:
            lengths.extend(mask.sum(dim=1).squeeze(1).numpy().tolist())
        lengths = np.array(lengths, dtype=np.int32)
        for i in range(samples.shape[0]):
            target_len = int(np.random.choice(lengths))
            samples[i, 0, target_len:] = 0.0

    out_path = os.path.join(cfg.out_dir, "samples_latest.npy")
    np.save(out_path, samples)
    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    main()
