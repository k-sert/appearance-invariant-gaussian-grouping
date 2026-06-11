# Appearance-Invariant Gaussian Grouping

This repository contains the code and results for combining **Gaussian Grouping** (GG) with **Gaussian Splatting in the Wild** (GS-W). The project investigates whether object-level identity encodings learned by Gaussian Grouping can improve appearance-invariant 3D Gaussian reconstruction under appearance changes and transient scene content.

Repository: <https://github.com/k-sert/Appearance-Invariant-Gaussian-Grouping>

## Project Structure

The GitHub repository is organized as:

```text
Appearance-Invariant-Gaussian-Grouping/
├── dataset_curation/    # Dataset split and appearance-variation notebook
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

Synthetic dataset preparation utilities are kept in:

```text
dataset_curation/data_curation.ipynb
```

## Method Overview

The project is organized into three stages.

### Stage I: Identity Transfer

Stage I trains Gaussian Grouping source models, extracts per-Gaussian identity encodings, and transfers them into the GS-W Gaussian space by nearest-neighbor matching in 3D. The transferred 16-dimensional identity features are then used as trainable semantic conditioning features inside GS-W.

Selected GG checkpoints used for identity transfer:

| Scene | GG experiment | GG source iteration |
|---|---|---:|
| figurines | figurines_phase_1 | 8000 |
| figurines_varied | figurines_phase_2 | 7000 |
| ramen | ramen_phase_1 | 7000 |
| ramen_varied | ramen_phase_2 | 7000 |
| teatime | teatime_phase_1 | 7000 |
| teatime_varied | teatime_phase_2 | 7000 |

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

## Dataset Curation

The repository includes a notebook for preparing the datasets used in the experiments:

```text
dataset_curation/data_curation.ipynb
```

The notebook creates GS-W-compatible train/test split files by assigning every 8th image to the test split. It also creates the appearance-varied datasets by copying the original scenes and applying synthetic appearance changes to 40% of the training images while keeping the test images unchanged.

Before running the notebook, update the dataset root near the top:

```python
DATASETS_ROOT = Path("../datasets").expanduser().resolve()
```

The optional COLMAP check section also exposes:

```python
GSW_ROOT = Path("../Stage-1/Gaussian-in-the-Wild").expanduser().resolve()
```

Change this path only if your local GS-W checkout is stored somewhere else.

## Running Stage I

From the repository root:

```bash
cd Stage-1
```

Use the GG environment for the Gaussian Grouping steps and the GS-W environment for the GS-W steps.

Train the Gaussian Grouping source model:

```bash
cd Gaussian-Grouping

python train.py \
  -s /path/to/datasets/figurines \
  -m /path/to/gg_outputs/figurines_phase_1 \
  --config_file config/gaussian_dataset/train.json \
  --iterations 30000 \
  --test_iterations 3000 4000 5000 6000 7000 8000 10000 12000 14000 16000 18000 20000 22500 25000 27500 30000 \
  --save_iterations 3000 4000 5000 6000 7000 8000 10000 12000 14000 16000 18000 20000 22500 25000 27500 30000
```

Extract the 16D GG identity vectors and Gaussian centers from saved GG checkpoints:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from plyfile import PlyData

gg_output = Path("/path/to/gg_outputs/figurines_phase_1")
iterations = [
    3000, 4000, 5000, 6000, 7000, 8000,
    10000, 12000, 14000, 16000, 18000, 20000,
    22500, 25000, 27500, 30000,
]

for iteration in iterations:
    iter_dir = gg_output / "point_cloud" / f"iteration_{iteration}"
    ply_path = iter_dir / "point_cloud.ply"
    if not ply_path.exists():
        continue

    vertex = PlyData.read(ply_path)["vertex"]
    identity = np.stack([vertex[f"obj_dc_{i}"] for i in range(16)], axis=1)
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    np.save(iter_dir / "identity_encodings.npy", identity)
    np.save(iter_dir / "gaussian_xyz.npy", xyz)
    print(f"Saved iteration {iteration}: xyz={xyz.shape}, identity={identity.shape}")
PY
```

Train a GS-W baseline for the same scene. This baseline is used for comparison and for probing which GG checkpoint transfers best:

```bash
cd ../Gaussian-in-the-Wild

python train.py \
  -s /path/to/datasets/figurines \
  -m /path/to/gsw_outputs/figurines_baseline_phase_1 \
  --scene_name figurines \
  --resolution 2 \
  --iterations 70000 \
  --test_iterations 70000 \
  --save_iterations 70000 \
  --eval
```

Run Stage I identity-conditioned GS-W training:

```bash
python train.py \
  -s /path/to/datasets/figurines \
  -m /path/to/gsw_outputs/figurines_finetuned_identity_phase_1 \
  --scene_name figurines \
  --iterations 70000 \
  --test_iterations 70000 \
  --save_iterations 70000 \
  --resolution 2 \
  --eval \
  --use_identity \
  --identity_dim 16 \
  --identity_trainable \
  --identity_path /path/to/gg_outputs/figurines_phase_1/point_cloud/iteration_8000/identity_encodings.npy \
  --identity_xyz_path /path/to/gg_outputs/figurines_phase_1/point_cloud/iteration_8000/gaussian_xyz.npy
```

Use the checkpoint table in the Stage I overview to swap `scene`, output names, GG experiment names, and GG iterations for the other final runs. The uploaded Stage I notebook also contains the optional checkpoint-probing code used to rank GG checkpoints against the trained GS-W baseline.

## Stage I Hyperparameters

Default final-run settings:

| Parameter | Value |
|---|---:|
| GG source training iterations | 30000 |
| GG save/test checkpoint grid | 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22500, 25000, 27500, 30000 |
| GG config file | `config/gaussian_dataset/train.json` |
| GG object classes | 256 |
| GG densify until iteration | 10000 |
| GG 3D regularization interval | 5 |
| GG 3D regularization k | 5 |
| GS-W baseline iterations | 70000 |
| Stage I GS-W iterations | 70000 |
| Resolution factor | 2 |
| Identity dimension | 16 |
| Identity transfer method | 1-nearest-neighbor in 3D |
| Identity trainable | true |
| Identity learning rate | 0.0025 |
| Checkpoint probe k-neighbors | 8 |

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
