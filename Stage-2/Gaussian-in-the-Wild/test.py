import subprocess

import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

# Commented out IPython magic to ensure Python compatibility.
from plyfile import PlyData
from codecarbon import EmissionsTracker
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizer
import numpy as np
import os

print("✅ GS-W setup complete")

"""## Baseline Experiments

The GS-W model is trained on all datasets to provide a performance baseline. Also, the energy consumption during training is tracked to estimate computational cost and emissions.
"""

DATASET_DIR = "/scratch-shared/gpuuva074/datasets"
BASE_OUTPUT = f"{os.path.dirname(os.path.abspath(__file__))}/output"


def train_GSW_base_model(scene, output_name, iterations=70000, resolution=2):
    SOURCE = f"{DATASET_DIR}/{scene}"
    OUTPUT = f"{BASE_OUTPUT}/{output_name}"

    tracker = EmissionsTracker(
        project_name=f"GSW_{output_name}", measure_power_secs=300
    )
    tracker.start()

    try:
        subprocess.run(
            [
                "python",
                "train.py",
                "-s",
                SOURCE,
                "-m",
                OUTPUT,
                "--scene_name",
                scene,
                "--resolution",
                str(resolution),
                "--iterations",
                str(iterations),
                "--test_iterations",
                str(iterations),
                "--save_iterations",
                str(iterations),
                "--eval",
            ],
            check=True,
        )
    finally:
        emissions = tracker.stop()
        print(f"Estimated CO2 emissions: {emissions:.6f} kg")


train_GSW_base_model("figurines", "figurines_baseline_phase_1")

train_GSW_base_model("ramen", "ramen_baseline_phase_1")

train_GSW_base_model("teatime", "teatime_baseline_phase_1")

# train_GSW_base_model("figurines_varied", "figurines_baseline_phase_2")
#
# train_GSW_base_model("ramen_varied", "ramen_baseline_phase_2")
#
# train_GSW_base_model("teatime_varied", "teatime_baseline_phase_2")

"""## Identity Coverage Selection

This section selects the Gaussian Grouping checkpoint whose learned identity encodings best align with the baseline GS-W Gaussians.

For each Gaussian Grouping checkpoint, we compare its Gaussian XYZ positions with the Gaussian centers produced by the baseline GS-W model. The goal is to estimate how well each checkpoint covers the GS-W geometry and how smoothly its identity features transfer to the GS-W Gaussians.

The checkpoint with the best combined coverage and identity-consistency score is selected. Its identity encodings and corresponding XYZ positions are then passed to GS-W for training with identity features.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from plyfile import PlyData


# -----------------------------
# 1. Load GS-W Gaussian centers
# -----------------------------
def load_ply_xyz(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return xyz.astype(np.float32)


# -----------------------------
# 2. Normalize identity vectors
# -----------------------------
def normalize_features(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


# -----------------------------
# 3. Probe one GG checkpoint
# -----------------------------
def probe_one_checkpoint(
    gg_xyz_path,
    gg_identity_path,
    gsw_ply_path,
    k_smooth=8,
    coverage_quantile=0.95,
):
    # Load data
    gg_xyz = np.load(gg_xyz_path).astype(np.float32)
    gg_id = np.load(gg_identity_path).astype(np.float32)
    gsw_xyz = load_ply_xyz(gsw_ply_path)

    assert gg_xyz.shape[0] == gg_id.shape[0], "GG xyz and identity count mismatch"

    # ---------------------------------------------------
    # A. Geometric compatibility: GS-W -> nearest GG
    # ---------------------------------------------------
    gg_tree = cKDTree(gg_xyz)
    nn_dist, nn_idx = gg_tree.query(gsw_xyz, k=1)

    transferred_id = gg_id[nn_idx]

    # ---------------------------------------------------
    # B. Symmetric distance: GG -> nearest GS-W
    # ---------------------------------------------------
    gsw_tree = cKDTree(gsw_xyz)
    gg_to_gsw_dist, _ = gsw_tree.query(gg_xyz, k=1)

    chamfer_approx = float(np.mean(nn_dist) + np.mean(gg_to_gsw_dist))

    # ---------------------------------------------------
    # C. Identity smoothness on GS-W KNN graph
    # ---------------------------------------------------
    _, gsw_knn_idx = gsw_tree.query(gsw_xyz, k=k_smooth + 1)
    gsw_knn_idx = gsw_knn_idx[:, 1:]  # remove self

    center_id = transferred_id
    neigh_id = transferred_id[gsw_knn_idx]

    center_norm = normalize_features(center_id)
    neigh_norm = normalize_features(neigh_id)

    cosine_sim = np.sum(center_norm[:, None, :] * neigh_norm, axis=-1)
    identity_l2 = np.linalg.norm(center_id[:, None, :] - neigh_id, axis=-1)

    mean_cosine = float(np.mean(cosine_sim))
    median_cosine = float(np.median(cosine_sim))
    mean_identity_l2 = float(np.mean(identity_l2))
    median_identity_l2 = float(np.median(identity_l2))

    # ---------------------------------------------------
    # D. Local identity variance
    # ---------------------------------------------------
    local_var = np.var(neigh_id, axis=1).mean(axis=1)

    # ---------------------------------------------------
    # E. Coverage threshold
    # Use GG internal spacing as scale reference
    # ---------------------------------------------------
    gg_self_dist, _ = gg_tree.query(gg_xyz, k=2)
    gg_spacing = gg_self_dist[:, 1]

    threshold = 2.0 * np.median(gg_spacing)
    percent_bad_coverage = float(np.mean(nn_dist > threshold) * 100.0)

    # ---------------------------------------------------
    # F. Distance-weighted identity interpolation
    # Optional soft transfer sanity check
    # ---------------------------------------------------
    soft_k = min(4, len(gg_xyz))
    soft_dist, soft_idx = gg_tree.query(gsw_xyz, k=soft_k)

    weights = 1.0 / (soft_dist + 1e-8)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    soft_id = np.sum(gg_id[soft_idx] * weights[..., None], axis=1)

    soft_center_norm = normalize_features(soft_id)
    soft_neigh_norm = normalize_features(soft_id[gsw_knn_idx])

    soft_cosine = np.sum(soft_center_norm[:, None, :] * soft_neigh_norm, axis=-1)

    soft_identity_l2 = np.linalg.norm(
        soft_id[:, None, :] - soft_id[gsw_knn_idx], axis=-1
    )

    return {
        "num_gg": len(gg_xyz),
        "num_gsw": len(gsw_xyz),
        "mean_nn_dist": float(np.mean(nn_dist)),
        "median_nn_dist": float(np.median(nn_dist)),
        "p90_nn_dist": float(np.percentile(nn_dist, 90)),
        "p95_nn_dist": float(np.percentile(nn_dist, 95)),
        "max_nn_dist": float(np.max(nn_dist)),
        "mean_gg_to_gsw_dist": float(np.mean(gg_to_gsw_dist)),
        "p95_gg_to_gsw_dist": float(np.percentile(gg_to_gsw_dist, 95)),
        "chamfer_approx": chamfer_approx,
        "coverage_threshold": float(threshold),
        "percent_bad_coverage": percent_bad_coverage,
        "mean_identity_knn_cosine": mean_cosine,
        "median_identity_knn_cosine": median_cosine,
        "mean_identity_knn_l2": mean_identity_l2,
        "median_identity_knn_l2": median_identity_l2,
        "mean_local_identity_variance": float(np.mean(local_var)),
        "soft_mean_identity_knn_cosine": float(np.mean(soft_cosine)),
        "soft_mean_identity_knn_l2": float(np.mean(soft_identity_l2)),
    }


# -----------------------------
# 4. Probe multiple GG checkpoints
# -----------------------------
def probe_multiple_checkpoints(
    gg_root,
    gg_iters,
    gsw_ply_path,
    k_smooth=8,
):
    rows = []

    for it in gg_iters:
        gg_base = os.path.join(gg_root, f"iteration_{it}")

        gg_xyz_path = os.path.join(gg_base, "gaussian_xyz.npy")
        gg_identity_path = os.path.join(gg_base, "identity_encodings.npy")

        if not os.path.exists(gg_xyz_path):
            print(f"Missing xyz for GG{it}: {gg_xyz_path}")
            continue

        if not os.path.exists(gg_identity_path):
            print(f"Missing identity for GG{it}: {gg_identity_path}")
            continue

        print(f"Probing GG{it}...")

        result = probe_one_checkpoint(
            gg_xyz_path=gg_xyz_path,
            gg_identity_path=gg_identity_path,
            gsw_ply_path=gsw_ply_path,
            k_smooth=k_smooth,
        )

        result["GG iteration"] = it
        rows.append(result)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No checkpoints were successfully probed.")

    df = df[
        [
            "GG iteration",
            "num_gg",
            "num_gsw",
            "mean_nn_dist",
            "median_nn_dist",
            "p90_nn_dist",
            "p95_nn_dist",
            "max_nn_dist",
            "mean_gg_to_gsw_dist",
            "p95_gg_to_gsw_dist",
            "chamfer_approx",
            "coverage_threshold",
            "percent_bad_coverage",
            "mean_identity_knn_cosine",
            "median_identity_knn_cosine",
            "mean_identity_knn_l2",
            "median_identity_knn_l2",
            "mean_local_identity_variance",
            "soft_mean_identity_knn_cosine",
            "soft_mean_identity_knn_l2",
        ]
    ]

    return df


# -----------------------------
# 5. Ranking score
# -----------------------------
def add_probe_score(df):
    """
    Lower score = better.
    Combines:
    - p95 NN distance
    - bad coverage %
    - identity L2
    - negative cosine similarity
    """

    scored = df.copy()

    def minmax(x):
        x = np.asarray(x, dtype=np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    scored["score_p95_nn"] = minmax(scored["p95_nn_dist"])
    scored["score_bad_coverage"] = minmax(scored["percent_bad_coverage"])
    scored["score_identity_l2"] = minmax(scored["mean_identity_knn_l2"])
    scored["score_cosine"] = minmax(1.0 - scored["mean_identity_knn_cosine"])

    scored["combined_score"] = (
        0.35 * scored["score_p95_nn"]
        + 0.25 * scored["score_bad_coverage"]
        + 0.25 * scored["score_identity_l2"]
        + 0.15 * scored["score_cosine"]
    )

    scored = scored.sort_values("combined_score", ascending=True)

    return scored


scene = "figurines_phase_1"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/figurines_baseline_phase_1/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

scene = "figurines_phase_2"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/figurines_baseline_phase_2/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

scene = "ramen_phase_1"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/ramen_baseline_phase_1/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

scene = "ramen_phase_2"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/ramen_baseline_phase_2/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

scene = "teatime_phase_1"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/teatime_baseline_phase_1/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

scene = "teatime_phase_2"

gg_root = f"{BASE_OUTPUT}/{scene}/point_cloud"
gsw_ply_path = f"{BASE_OUTPUT}/teatime_baseline_phase_2/ckpts_point_cloud/iteration_70000/point_cloud.ply"

gg_iters = [
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22500,
    25000,
    27500,
    30000,
]

df = probe_multiple_checkpoints(
    gg_root=gg_root,
    gg_iters=gg_iters,
    gsw_ply_path=gsw_ply_path,
    k_smooth=8,
)

scored_df = add_probe_score(df)

scored_df

"""## Identity Feature Experiments

The best-performing checkpoints from the Gaussian Grouping model are used as input to the GS-W model. The training configuration follows the original GS-W setup as provided by the authors.

Identity features are set to be trainable (fine-tuned). Prior ablation studies showed that fine-tuning is necessary, especially for longer GS-W training schedules, to achieve higher performance. Therefore, experiments with frozen identity encodings are not included.

Finally, the energy consumption during training is tracked to estimate computational cost and emissions.
"""


def train_GSW_identity_model(
    scene,
    output_name,
    gg_experiment,
    gg_iteration,
    identity_trainable=False,
    iterations=70000,
    resolution=2,
):
    SOURCE = f"{DATASET_DIR}/{scene}"
    OUTPUT = f"{BASE_OUTPUT}/{output_name}"

    GG_POINT_CLOUD = (
        f"{BASE_OUTPUT}/{gg_experiment}/point_cloud/iteration_{gg_iteration}"
    )

    ID_PATH = f"{GG_POINT_CLOUD}/identity_encodings.npy"
    XYZ_PATH = f"{GG_POINT_CLOUD}/gaussian_xyz.npy"

    tracker = EmissionsTracker(project_name=f"GSW_{output_name}")
    tracker.start()

    try:
        cmd = [
            "python",
            "train.py",
            "-s",
            SOURCE,
            "-m",
            OUTPUT,
            "--scene_name",
            scene,
            "--iterations",
            str(iterations),
            "--test_iterations",
            str(iterations),
            "--save_iterations",
            str(iterations),
            "--resolution",
            str(resolution),
            "--eval",
            "--use_identity",
            "--identity_dim",
            "16",
            "--identity_path",
            ID_PATH,
            "--identity_xyz_path",
            XYZ_PATH,
        ]
        if identity_trainable:
            cmd.append("--identity_trainable")
        subprocess.run(cmd, check=True)
    finally:
        emissions = tracker.stop()
        print(f"Estimated CO2 emissions: {emissions:.6f} kg")


train_GSW_identity_model(
    scene="figurines",
    output_name="figurines_finetuned_identity_phase_1",
    gg_experiment="figurines_phase_1",
    gg_iteration=8000,
    identity_trainable=True,
)

# train_GSW_identity_model(
#     scene="figurines_varied",
#     output_name="figurines_finetuned_identity_phase_2",
#     gg_experiment="figurines_phase_2",
#     gg_iteration=7000,
#     identity_trainable=True,
# )

train_GSW_identity_model(
    scene="ramen",
    output_name="ramen_finetuned_identity_phase_1",
    gg_experiment="ramen_phase_1",
    gg_iteration=7000,
    identity_trainable=True,
)

# train_GSW_identity_model(
#     scene="ramen_varied",
#     output_name="ramen_finetuned_identity_phase_2",
#     gg_experiment="ramen_phase_2",
#     gg_iteration=7000,
#     identity_trainable=True,
# )

train_GSW_identity_model(
    scene="teatime",
    output_name="teatime_finetuned_identity_phase_1",
    gg_experiment="teatime_phase_1",
    gg_iteration=7000,
    identity_trainable=True,
)

# train_GSW_identity_model(
#     scene="teatime_varied",
#     output_name="teatime_finetuned_identity_phase_2",
#     gg_experiment="teatime_phase_2",
#     gg_iteration=7000,
#     identity_trainable=True,
# )

"""## Performance Results

The performance metrics (PSNR, SSIM, LPIPS) from all Stage 2 experiments are aggregated and processed here for further visualization.
"""

import os
import json
import pandas as pd

BASE_DIR = BASE_OUTPUT
BASE_OUTPUT = f"{os.path.dirname(os.path.abspath(__file__))}/output"


def collect_results():
    phase_1_rows = []
    phase_2_rows = []

    for exp_name in os.listdir(BASE_DIR):
        exp_path = os.path.join(BASE_DIR, exp_name)

        if not os.path.isdir(exp_path):
            continue

        json_path = os.path.join(exp_path, "results.json")

        if not os.path.exists(json_path):
            print(f"Skipping (no results.json): {exp_name}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        # Expecting something like "ours_70000"
        result_key = list(data.keys())[0]
        metrics = data[result_key]

        row = {
            "experiment": exp_name,
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "LPIPS": metrics.get("LPIPS"),
        }

        # Split into phases
        if "phase_1" in exp_name:
            phase_1_rows.append(row)
        elif "phase_2" in exp_name:
            phase_2_rows.append(row)
        else:
            print(f"Unknown phase in: {exp_name}")

    df_phase_1 = pd.DataFrame(phase_1_rows)
    df_phase_2 = pd.DataFrame(phase_2_rows)

    return df_phase_1, df_phase_2


df_phase_1_clean, df_phase_2_varied = collect_results()

print(df_phase_1_clean)
print(df_phase_2_varied)

import matplotlib.pyplot as plt


def plot_phase_metric_bars(df, phase_name):
    # df = prepare_phase_comparison(df)

    datasets = ["figurines", "ramen", "teatime"]
    models = ["baseline", "finetuned_identity"]
    metrics = ["PSNR", "SSIM", "LPIPS"]

    for metric in metrics:
        values = (
            df.pivot(index="dataset", columns="model", values=metric)
            .reindex(datasets)
            .reindex(columns=models)
        )

        x = np.arange(len(datasets))
        width = 0.18  # 🔥 thinner bars

        plt.figure(figsize=(6.5, 4.5))

        plt.bar(
            x - width / 2,
            values["baseline"],
            width,
            label="Baseline",
            color="silver",
            edgecolor="black",
            linewidth=0.5,
        )

        plt.bar(
            x + width / 2,
            values["finetuned_identity"],
            width,
            label="Finetuned identity",
            color="tab:orange",
            edgecolor="black",
            linewidth=0.5,
        )

        plt.xticks(x, datasets)
        plt.ylabel(metric)
        plt.title(f"{phase_name} - {metric}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        # 🔍 Zoom logic
        min_val = values.min().min()
        max_val = values.max().max()

        margin = (max_val - min_val) * 0.4 if max_val != min_val else 0.01

        plt.ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.show()


plot_phase_metric_bars(df_phase_1_clean, "Original")
# plot_phase_metric_bars(df_phase_2_varied, "Appearance-varied")
