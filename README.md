# Appearance-Invariant Gaussian Grouping

This repository contains the code and results for combining **Gaussian Grouping** (GG) with **Gaussian Splatting in the Wild** (GS-W). The project investigates whether object-level identity encodings learned by Gaussian Grouping can improve appearance-invariant 3D Gaussian reconstruction under appearance changes and transient scene content.

Repository: <https://github.com/k-sert/Appearance-Invariant-Gaussian-Grouping>

## Project Structure

The GitHub repository is organized as:

```text
Appearance-Invariant-Gaussian-Grouping/
├── env_scripts/         # Optional environment setup and activation helpers
├── results/             # Baseline and Stage specific results
├── Stage-1/             # Identity transfer experiments
├── Stage-2/             # Static-dynamic routing experiments
├── Stage-3/             # Boundary-aware regularization experiments
└── README.md
```

The main stage folders contain their own modified copies of the upstream repositories:

```text
Stage-1/
├── Gaussian-Grouping/
└── Gaussian-in-the-Wild/

Stage-2/
├── Gaussian-Grouping/
└── Gaussian-in-the-Wild/

Stage-3/
├── boundary_metrics/
├── Gaussian-Grouping/
├── Gaussian-in-the-Wild/
└── stage3_boundary_regularizer.py
```

The original upstream READMEs are available inside the corresponding GG and GS-W folders.

## Method Overview

The project is organized into three stages.

### Stage I: Identity Transfer

Stage I trains Gaussian Grouping source models, extracts per-Gaussian identity encodings, and transfers them into the GS-W Gaussian space by nearest-neighbor matching in 3D. The transferred 16-dimensional identity features are then used as trainable semantic conditioning features inside GS-W.

Selected GG checkpoints used for identity transfer:

| Dataset | GG source iteration |
|---|---:|
| Figurines | 8000 |
| Figurines-synthetic | 7000 |
| Ramen | 7000 |
| Ramen-synthetic | 7000 |
| Teatime | 7000 |
| Teatime-synthetic | 7000 |

Relevant code:

```text
Stage-1/Gaussian-Grouping/
Stage-1/Gaussian-in-the-Wild/
```

### Stage II: Static-Dynamic Routing

Stage II builds on the Stage I identity-conditioned GS-W model. It adds an identity-conditioned MLP that predicts a per-Gaussian dynamic probability:

```text
16D identity -> Linear(16, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
```

The output `p` gates the appearance branches:

- static branch gate: `1 - p`
- dynamic branch gate: `p`

The routing MLP is optimized end-to-end through the reconstruction objective. The final implementation does not use an additional explicit routing-supervision loss.

Relevant code:

```text
Stage-2/Gaussian-in-the-Wild/net_modules/basic_mlp.py
Stage-2/Gaussian-in-the-Wild/scene/gaussian_model.py
```

### Stage III: Boundary-Aware Regularization

Stage III adds boundary-aware regularization to improve local object separation near segmentation boundaries. Boundary maps are generated from instance masks, the rasterizer renders an identity-aware feature field, and boundary losses are applied to the rendered identity features.

Relevant code:

```text
Stage-3/stage3_boundary_regularizer.py
Stage-3/Gaussian-in-the-Wild/
Stage-3/boundary_metrics/
```

## Environment Setup

Use any Python/conda environment that can build and import the CUDA extensions used by Gaussian Grouping and GS-W. The `env_scripts/` folder contains helper scripts used during development, but the project does not require a specific cluster or scheduler.

```bash
cd env_scripts
chmod +x setup_gg_env.sh setup_gsw_env.sh activate_gg.sh activate_gsw.sh
```

Create the GG environment:

```bash
./setup_gg_env.sh
```

Create the GS-W environment:

```bash
./setup_gsw_env.sh
```

Activate an environment in a new terminal/session:

```bash
source env_scripts/activate_gg.sh
# or
source env_scripts/activate_gsw.sh
```

If you use your own environment, verify that PyTorch and the CUDA extensions import correctly:

```bash
python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

## Data Layout

Use any local or shared filesystem layout and pass the paths explicitly to the scripts. The examples below use placeholder paths:

```text
/path/to/datasets
```

and precomputed Gaussian Grouping outputs:

```text
/path/to/gg_outputs
```

Expected GG output structure:

```text
gg_outputs/
├── figurines_phase_1/
│   └── point_cloud/iteration_8000/
│       ├── identity_encodings.npy
│       └── gaussian_xyz.npy
├── ramen_phase_1/
│   └── point_cloud/iteration_7000/
└── ...
```

Scenes used in the final evaluation:

```text
figurines
ramen
teatime
figurines_varied
ramen_varied
teatime_varied
```

Stage III generates boundary data inside each scene folder:

```text
<dataset>/<scene>/
├── boundary_confidence/
├── boundary_instance/
└── boundary_vis/
```

## Running Stage III

From the repository root:

```bash
cd Stage-3
```

Generate boundary maps:

```bash
python stage3_boundary_regularizer.py generate-boundaries \
  --datasets-root /path/to/datasets \
  --scene figurines \
  --edge-width 3
```

Run Stage III training:

```bash
python stage3_boundary_regularizer.py train \
  --gsw-root Gaussian-in-the-Wild \
  --datasets-root /path/to/datasets \
  --gg-root /path/to/gg_outputs \
  --scene figurines \
  --output-name figurines_stage3_70k \
  --gg-experiment figurines_phase_1 \
  --gg-iteration 8000 \
  --iterations 70000 \
  --resolution 2 \
  --boundary-loss-coef 0.01 \
  --boundary-footprint-loss-coef 0.002 \
  --boundary-edge-threshold 0.2 \
  --boundary-similarity-margin 0.15 \
  --boundary-max-points 4096 \
  --boundary-k-neighbors 8 \
  --boundary-identity-trainable \
  --use-codecarbon
```

The `--gg-root` argument accepts both layouts:

```text
Gaussian-Grouping/output/<experiment>/point_cloud/iteration_<iter>/
gg_outputs/<experiment>/point_cloud/iteration_<iter>/
```

## Stage III Hyperparameters

Default final-run settings:

| Parameter | Value |
|---|---:|
| Iterations | 70000 |
| Resolution factor | 2 |
| Identity dimension | 16 |
| Boundary identity trainable | true |
| Boundary identity loss coefficient | 0.01 |
| Boundary footprint loss coefficient | 0.002 |
| Boundary edge threshold | 0.2 |
| Boundary similarity margin | 0.15 |
| Boundary max points | 4096 |
| Boundary k-neighbors | 8 |
| Boundary-map edge width | 3 px |
| Boundary-band evaluation width | 5 px |

## Boundary Metrics

The public repository includes two Stage III analysis utilities:

```text
Stage-3/boundary_metrics/compute_boundary_band_metrics.py
Stage-3/boundary_metrics/generate_stage3_improvement_heatmaps.py
```

Compute boundary-band metrics:

```bash
python Stage-3/boundary_metrics/compute_boundary_band_metrics.py
```

Generate Stage III improvement heatmaps:

```bash
python Stage-3/boundary_metrics/generate_stage3_improvement_heatmaps.py
```

Both scripts contain path configuration near the top. Update those paths before running if your datasets or GS-W outputs live in a different location.

## Results

The repository-level `results/` folder contains the figures and result artifacts included with the project.

## Practical Notes

- Use `PYTHONNOUSERSITE=1` if Python accidentally imports packages from `~/.local`.
- When rebuilding CUDA extensions, set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` for your GPU.
- If `diff_gaussian_rasterization` imports from the wrong Python installation, check:

```bash
which python
python - <<'PY'
import sys, site
print(sys.executable)
print(site.ENABLE_USER_SITE)
print(sys.path)
PY
```

## References

This project builds on:

- [Gaussian Grouping: Segment and Edit Anything in 3D Scenes](https://github.com/lkeab/gaussian-grouping)
- [Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections](https://github.com/EastbeanZhang/Gaussian-Wild)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
