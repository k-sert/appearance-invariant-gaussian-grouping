"""
# Stage 1 - Gaussian Grouping

## Dependencies
"""

import subprocess

import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

from plyfile import PlyData
from codecarbon import EmissionsTracker
import diff_gaussian_rasterization
import simple_knn
import numpy as np
import os

print("✅ GG setup complete")

"""## Train the GG Model

The energy consumption during training is tracked to estimate computational cost and emissions.

### Phase 1 - Original Setting

In this phase, the original datasets of the Gaussian Grouping paper are used.
"""

DATASET_DIR = "/scratch-shared/gpuuva074/datasets"
BASE_OUTPUT = f"{os.path.dirname(os.path.abspath(__file__))}/output"


def train_GG_model(scene, output_name):
    SOURCE = f"{DATASET_DIR}/{scene}"
    OUTPUT = f"{BASE_OUTPUT}/{output_name}"
    CONFIG = f"{os.path.dirname(os.path.abspath(__file__))}/config/gaussian_dataset/train.json"

    iters = [
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

    tracker = EmissionsTracker(project_name=f"GG_{output_name}")
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
                "--config_file",
                CONFIG,
                "--iterations",
                "30000",
                "--test_iterations",
                *map(str, iters),
                "--save_iterations",
                *map(str, iters),
            ],
            check=True,
        )
    finally:
        emissions = tracker.stop()
        print(f"Estimated CO2 emissions: {emissions:.6f} kg")


train_GG_model("figurines", "figurines_phase_1")

train_GG_model("ramen", "ramen_phase_1")

train_GG_model("teatime", "teatime_phase_1")

# """### Phase 2 - Appearance-varied Setting
#
# In this phase, the modified datasets of the Gaussian Grouping paper are used. Please refer to `data_curation.ipynb` in the Drive folder for more details.
# """
#
# train_GG_model("figurines_varied", "figurines_phase_2")
#
# train_GG_model("ramen_varied", "ramen_phase_2")
#
# train_GG_model("teatime_varied", "teatime_phase_2")

"""## Extract Identity Encodings

Here, we store the learned identity encodings and corresponding XYZ positions for each checkpoint. Using these together with the baseline GS-W model, we can evaluate which checkpoint yields the best performance.
"""


# scenes = [
#     "figurines_phase_1",
#     "ramen_phase_1",
#     "teatime_phase_1",
#     "figurines_phase_2",
#     "ramen_phase_2",
#     "teatime_phase_2",
# ]
scenes = [
    "figurines_phase_1",
    "ramen_phase_1",
    "teatime_phase_1",
]

iterations = [
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

for scene in scenes:
    print(f"\n=== Processing scene: {scene} ===")

    scene_output = os.path.join(BASE_OUTPUT, scene)

    for iteration in iterations:
        iter_dir = os.path.join(scene_output, "point_cloud", f"iteration_{iteration}")
        ply_path = os.path.join(iter_dir, "point_cloud.ply")

        if not os.path.exists(ply_path):
            print(f"  [Skip] {scene} - iter {iteration}: no point_cloud.ply")
            continue

        point_cloud = PlyData.read(ply_path)
        v = point_cloud["vertex"]

        obj_keys = [f"obj_dc_{i}" for i in range(16)]

        # Safety check
        if not all(k in v.data.dtype.names for k in obj_keys):
            print(f"  [Skip] {scene} - iter {iteration}: missing identity keys")
            continue

        identity = np.stack([v[k] for k in obj_keys], axis=1)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)

        np.save(os.path.join(iter_dir, "identity_encodings.npy"), identity)
        np.save(os.path.join(iter_dir, "gaussian_xyz.npy"), xyz)

        print(
            f"  [OK] {scene} - iter {iteration} | xyz: {xyz.shape} | id: {identity.shape}"
        )
