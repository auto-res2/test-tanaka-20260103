"""src/main.py
Main orchestrator – loads run config, applies mode-specific overrides, and spawns train.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).parent.parent / "config"
RUNS_DIR = CONFIG_DIR / "runs"


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert "run" in cfg, "CLI arg run=<run_id> is required."
    run_id: str = cfg.run

    run_cfg_path = RUNS_DIR / f"{run_id}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_cfg_path}")

    base_cfg = OmegaConf.load(to_absolute_path(str(run_cfg_path)))
    merged = OmegaConf.merge(base_cfg, cfg)  # CLI overrides (mode, results_dir,…)

    # Mode-specific tweaks
    if merged.mode == "trial":
        merged.wandb.mode = "disabled"
        if "optuna" in merged:
            merged.optuna.n_trials = 0
        merged.training.epochs = 1
        merged.training.batch_size = min(2, merged.training.batch_size)
    elif merged.mode == "full":
        merged.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Persist final cfg for reproducibility & for train.py consumption
    results_dir = Path(to_absolute_path(merged.results_dir)).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    final_cfg_path = results_dir / f"{run_id}_final_cfg.yaml"
    OmegaConf.save(merged, final_cfg_path)

    # Spawn train.py as subprocess (delegates all heavy lifting)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"cfg_path={final_cfg_path.as_posix()}",
        f"mode={merged.mode}",
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
