import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(__file__)


def _parse_exp_name(exp_name: str):
    """Parse '{dataset}_{model}_phase_{n}' into (dataset, model, phase)."""
    match = re.search(r"_?(phase_[12])_?", exp_name)
    if not match:
        return None, None, None

    phase = match.group(1)
    remainder = exp_name[: match.start()].strip("_")

    parts = remainder.split("_", 1)
    dataset = parts[0]
    model = parts[1] if len(parts) > 1 else "ours"

    return dataset, model, phase


def collect_results(stage: str):
    phase_1_rows = []
    phase_2_rows = []

    stage_dir = os.path.join(BASE_DIR, stage)

    for exp_name in os.listdir(stage_dir):
        exp_path = os.path.join(stage_dir, exp_name)

        if not os.path.isdir(exp_path):
            continue

        json_path = os.path.join(exp_path, "results.json")

        if not os.path.exists(json_path):
            print(f"Skipping (no results.json): {exp_name}")
            continue

        dataset, model, phase = _parse_exp_name(exp_name)

        if phase is None:
            print(f"Unknown phase in: {exp_name}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        result_key = list(data.keys())[0]
        metrics = data[result_key]

        row = {
            "experiment": exp_name,
            "dataset": dataset,
            "model": model,
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "LPIPS": metrics.get("LPIPS"),
        }

        if phase == "phase_1":
            phase_1_rows.append(row)
        else:
            phase_2_rows.append(row)

    df_phase_1 = pd.DataFrame(phase_1_rows)
    df_phase_2 = pd.DataFrame(phase_2_rows)

    return df_phase_1, df_phase_2


def plot_phase_metric_bars(df, phase_name, out_dir: str):
    if df.empty:
        print(f"No data for {phase_name}, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    datasets = sorted(df["dataset"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    metrics = ["PSNR", "SSIM", "LPIPS"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n = len(models)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    for metric in metrics:
        values = (
            df.pivot(index="dataset", columns="model", values=metric)
            .reindex(datasets)
            .reindex(columns=models)
        )

        x = np.arange(len(datasets))

        plt.figure(figsize=(6.5, 4.5))

        for i, model in enumerate(models):
            plt.bar(
                x + offsets[i],
                values[model],
                width,
                label=model.replace("_", " ").title(),
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

        plt.xticks(x, datasets)
        plt.ylabel(metric)
        plt.title(f"{phase_name} - {metric}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        min_val = values.min().min()
        max_val = values.max().max()
        margin = (max_val - min_val) * 0.4 if max_val != min_val else 0.01
        plt.ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        filename = f"{phase_name.lower().replace(' ', '_')}_{metric.lower()}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def main(stage: str):
    if not os.path.exists(stage):
        raise ValueError(f"Could not fine stage: {stage}")

    phase_1_results, phase_2_results = collect_results(stage=stage)

    plots_dir = os.path.join(BASE_DIR, "plots", stage)
    plot_phase_metric_bars(phase_1_results, "Original", os.path.join(plots_dir, "phase_1"))
    plot_phase_metric_bars(phase_2_results, "Appearance-varied", os.path.join(plots_dir, "phase_2"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=str)
    args = parser.parse_args()

    main(stage=args.stage)
