"""src/evaluate.py
Independent evaluation & visualisation script.
Usage:
uv run python -m src.evaluate results_dir=PATH run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf


def export_metrics(run_id: str, history_df, summary_dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({"history": history_df.to_dict(orient="list"), "summary": summary_dict}, f, indent=2)


def plot_learning_curve(run_id: str, history_df, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    if "train_loss" in history_df:
        sns.lineplot(x=history_df.index, y=history_df["train_loss"], label="train_loss")
    if "val_loss" in history_df:
        sns.lineplot(x=history_df.index, y=history_df["val_loss"], label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    fname = f"{run_id}_learning_curve.pdf"
    plt.savefig(out_dir / fname)
    plt.close()
    print(out_dir / fname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list, e.g. '[\"run-1\", \"run-2\"]'")
    args = parser.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(Path(__file__).parent.parent / "config" / "config.yaml")
    entity, project = cfg.wandb.entity, cfg.wandb.project

    api = wandb.Api()
    primary_metric = "final_accuracy_at_budget"  # per hypothesis
    aggregated: Dict[str, Dict[str, float]] = {}
    run_primary: Dict[str, float] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history_df = run.history(keys=None)
        summary = run.summary._json_dict
        export_dir = results_root / run_id
        export_metrics(run_id, history_df, summary, export_dir)
        plot_learning_curve(run_id, history_df, export_dir)

        pm_val = summary.get(primary_metric) or summary.get("final_val_acc") or summary.get("val_acc")
        if pm_val is None:
            raise ValueError(f"Primary metric '{primary_metric}' not found for run {run_id}")
        run_primary[run_id] = float(pm_val)
        for k, v in summary.items():
            aggregated.setdefault(k, {})[run_id] = v

    # ---------------- aggregate comparison ----------------
    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    best_prop = max((r for r in run_primary if "proposed" in r), key=lambda x: run_primary[x])
    best_base = max((r for r in run_primary if any(t in r for t in ("comparative", "baseline"))), key=lambda x: run_primary[x])
    gap_pct = (run_primary[best_prop] - run_primary[best_base]) / run_primary[best_base] * 100.0

    aggregated_json = {
        "primary_metric": primary_metric,
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop, "value": run_primary[best_prop]},
        "best_baseline": {"run_id": best_base, "value": run_primary[best_base]},
        "gap": gap_pct,
    }
    with (comparison_dir / "aggregated_metrics.json").open("w") as f:
        json.dump(aggregated_json, f, indent=2)

    # Bar chart for primary metric
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(run_primary.keys()), y=list(run_primary.values()))
    plt.ylabel(primary_metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    chart_path = comparison_dir / "comparison_primary_metric_bar_chart.pdf"
    plt.savefig(chart_path)
    plt.close()
    print(chart_path)

    print("Aggregated metrics:", comparison_dir / "aggregated_metrics.json")


if __name__ == "__main__":
    main()
