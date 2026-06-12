# Results

Evaluation results (PSNR, SSIM, LPIPS) and plotting scripts for the Appearance-Invariant Gaussian Grouping experiments.

## Directory structure

```
results/
├── Baselines/                   # Vanilla Gaussian Grouping baseline
├── Stage-1/                     # Stage 1: baseline vs. finetuned model
├── Stage-1-combined/            # Stage 1 combined variant
├── Stage-2/                     # Stage 2: appearance-conditioned finetuning
├── Stage-2-v2/                  # Stage 2 v2: identity-preserving variant
├── Stage-3/                     # Stage 3: full pipeline (finetuned only)
└── plots/                       # Generated plots (created by plot_results.py)
```

Each stage directory contains experiment subdirectories named `{dataset}_{model}_phase_{n}`, e.g. `figurines_finetuned_phase_1`. Each subdirectory holds a `results.json` with per-run PSNR, SSIM, and LPIPS scores.

- **Datasets**: `figurines`, `ramen`, `teatime`
- **Models**: `baseline`, `finetuned`, `finetuned_identity`
- **Phase 1**: original appearance evaluation
- **Phase 2**: appearance-varied evaluation

## Generating plots

Dependencies are managed with `uv`. Run any of the commands below from this directory.

### Single stage

```sh
uv run plot_results.py --stage Stage-2
```

Outputs per-stage bar charts to `plots/Stage-2/phase_1/` and `plots/Stage-2/phase_2/`.

### Multiple stages (side-by-side comparison)

```sh
uv run plot_results.py --stage Stage-2 Baselines
```

Produces both per-stage plots and a cross-stage comparison under `plots/comparison/`.

```sh
uv run plot_results.py --stage Stage-1 Stage-1-combined
```

### Cross-stage comparison with a shared baseline

Use `--baseline` to pin one stage's baseline run as the reference bar in comparison plots. The baseline stage must be included in `--stage`.

```sh
uv run plot_results.py --stage Baselines Stage-1 Stage-2 Stage-3 --baseline Baselines
```

Without `--baseline`, all baseline-model runs are excluded from the comparison chart.
