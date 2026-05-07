"""
Stage 3: boundary-aware identity regularization for the GG -> GS-W pipeline.

1. Generate per-image boundary confidence maps from existing instance masks.
2. Launch GS-W training with transferred Gaussian Grouping identity features.

Example:
    python stage3_boundary_regularizer.py generate-boundaries \
        --datasets-root /path/to/datasets \
        --scene figurines \
        --edge-width 5

    python stage3_boundary_regularizer.py train \
        --gsw-root /path/to/Gaussian-Wild \
        --datasets-root /path/to/datasets \
        --gg-root /path/to/gaussian-grouping \
        --scene figurines \
        --output-name figurines_stage3_boundary_phase_1 \
        --gg-experiment figurines_phase_1 \
        --gg-iteration 8000 \
        --iterations 7000 \
        --resolution 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install opencv-python or opencv-python-headless.") from exc
    return cv2


def find_image_folder(scene_dir: Path) -> Path:
    if (scene_dir / "images").exists():
        return scene_dir / "images"
    image_files = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.jpeg")) + list(scene_dir.glob("*.png"))
    if image_files:
        return scene_dir
    raise FileNotFoundError(f"No image folder/images found under {scene_dir}")


def find_mask_folder(scene_dir: Path, explicit_mask_dir: str | None = None) -> Path:
    if explicit_mask_dir:
        mask_dir = Path(explicit_mask_dir)
        if not mask_dir.is_absolute():
            mask_dir = scene_dir / mask_dir
        if not mask_dir.exists():
            raise FileNotFoundError(mask_dir)
        return mask_dir

    candidates = [
        "object_mask",
        "object_masks",
        "masks",
        "mask",
        "segmentation",
        "segmentations",
        "sam_deva_masks",
        "deva_masks",
        "Annotations",
        "annotations",
    ]
    for name in candidates:
        path = scene_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No mask folder found under {scene_dir}. Tried: {candidates}")


def read_instance_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.array(Image.open(path))

    if mask.ndim == 3:
        flat = mask.reshape(-1, mask.shape[-1])
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        mask = inv.reshape(mask.shape[:2])

    return mask.astype(np.int64)


def locate_mask_for_image(mask_dir: Path, image_path: Path) -> Path:
    stem = image_path.stem
    for ext in [".npy", ".png", ".jpg", ".jpeg"]:
        path = mask_dir / f"{stem}{ext}"
        if path.exists():
            return path

    matches = sorted(mask_dir.glob(f"**/{stem}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No mask found for {image_path.name} in {mask_dir}")


def compute_boundary_confidence(mask: np.ndarray, edge_width: int) -> np.ndarray:
    cv2 = require_cv2()
    mask = mask.astype(np.int64)
    edge = np.zeros(mask.shape, dtype=np.uint8)

    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]

    if edge_width > 1:
        kernel = np.ones((edge_width, edge_width), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)

    dist = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 3)
    confidence = np.exp(-dist / max(edge_width, 1)).astype(np.float32)
    confidence[edge > 0] = 1.0
    return confidence


def generate_boundary_maps(args: argparse.Namespace) -> None:
    cv2 = require_cv2()
    scene_dir = Path(args.datasets_root) / args.scene
    image_dir = find_image_folder(scene_dir)
    mask_dir = find_mask_folder(scene_dir, args.mask_dir)

    conf_dir = scene_dir / args.boundary_confidence_dir
    inst_dir = scene_dir / args.boundary_instance_dir
    vis_dir = scene_dir / args.boundary_vis_dir
    conf_dir.mkdir(exist_ok=True)
    inst_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"])
    print(f"[Stage 3] Scene: {args.scene}")
    print(f"[Stage 3] Images: {len(image_paths)} from {image_dir}")
    print(f"[Stage 3] Masks: {mask_dir}")

    for image_path in image_paths:
        conf_path = conf_dir / f"{image_path.stem}.npy"
        inst_path = inst_dir / f"{image_path.stem}.npy"
        if conf_path.exists() and inst_path.exists() and not args.overwrite:
            continue

        mask_path = locate_mask_for_image(mask_dir, image_path)
        mask = read_instance_mask(mask_path)
        image_size = Image.open(image_path).size
        if mask.shape[::-1] != image_size:
            cv2 = require_cv2()
            mask = cv2.resize(mask.astype(np.int32), image_size, interpolation=cv2.INTER_NEAREST).astype(np.int64)

        confidence = compute_boundary_confidence(mask, edge_width=args.edge_width)
        np.save(conf_path, confidence.astype(np.float32))
        np.save(inst_path, mask.astype(np.int64))
        cv2.imwrite(str(vis_dir / f"{image_path.stem}.png"), (confidence * 255).astype(np.uint8))

    print(f"[Stage 3] Wrote boundary maps to {conf_dir}")


def train(args: argparse.Namespace) -> None:
    gsw_root = Path(args.gsw_root).resolve()
    datasets_root = Path(args.datasets_root).resolve()
    gg_root = Path(args.gg_root).resolve()
    source = datasets_root / args.scene
    output = gsw_root / "output" / args.output_name
    gg_point_cloud = gg_root / "output" / args.gg_experiment / "point_cloud" / f"iteration_{args.gg_iteration}"
    identity_path = gg_point_cloud / "identity_encodings.npy"
    xyz_path = gg_point_cloud / "gaussian_xyz.npy"
    confidence_path = source / args.boundary_confidence_dir
    instance_path = source / args.boundary_instance_dir

    for path in [gsw_root / "train.py", source, identity_path, xyz_path, confidence_path, instance_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(source),
        "-m",
        str(output),
        "--scene_name",
        args.scene,
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--save_iterations",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
        "--eval",
        "--use_identity",
        "--identity_dim",
        str(args.identity_dim),
        "--identity_trainable",
        "--identity_path",
        str(identity_path),
        "--identity_xyz_path",
        str(xyz_path),
        "--use_boundary_loss",
        "--boundary_identity_trainable",
        "--boundary_confidence_path",
        str(confidence_path),
        "--boundary_instance_path",
        str(instance_path),
        "--boundary_loss_coef",
        str(args.boundary_loss_coef),
        "--boundary_footprint_loss_coef",
        str(args.boundary_footprint_loss_coef),
        "--boundary_edge_threshold",
        str(args.boundary_edge_threshold),
        "--boundary_similarity_margin",
        str(args.boundary_similarity_margin),
        "--boundary_max_points",
        str(args.boundary_max_points),
        "--boundary_k_neighbors",
        str(args.boundary_k_neighbors),
    ]

    print("[Stage 3] Launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=gsw_root, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-boundaries", help="Create boundary maps from instance masks.")
    gen.add_argument("--datasets-root", required=True)
    gen.add_argument("--scene", required=True)
    gen.add_argument("--mask-dir", default=None, help="Mask directory, absolute or relative to the scene folder.")
    gen.add_argument("--edge-width", type=int, default=5)
    gen.add_argument("--boundary-confidence-dir", default="boundary_confidence")
    gen.add_argument("--boundary-instance-dir", default="boundary_instance")
    gen.add_argument("--boundary-vis-dir", default="boundary_vis")
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=generate_boundary_maps)

    tr = subparsers.add_parser("train", help="Run Stage 3 GS-W training.")
    tr.add_argument("--gsw-root", required=True)
    tr.add_argument("--datasets-root", required=True)
    tr.add_argument("--gg-root", required=True)
    tr.add_argument("--scene", required=True)
    tr.add_argument("--output-name", required=True)
    tr.add_argument("--gg-experiment", required=True)
    tr.add_argument("--gg-iteration", type=int, required=True)
    tr.add_argument("--iterations", type=int, default=70000)
    tr.add_argument("--resolution", type=int, default=2)
    tr.add_argument("--identity-dim", type=int, default=16)
    tr.add_argument("--boundary-confidence-dir", default="boundary_confidence")
    tr.add_argument("--boundary-instance-dir", default="boundary_instance")
    tr.add_argument("--boundary-loss-coef", type=float, default=0.05)
    tr.add_argument("--boundary-footprint-loss-coef", type=float, default=0.01)
    tr.add_argument("--boundary-edge-threshold", type=float, default=0.2)
    tr.add_argument("--boundary-similarity-margin", type=float, default=0.25)
    tr.add_argument("--boundary-max-points", type=int, default=4096)
    tr.add_argument("--boundary-k-neighbors", type=int, default=8)
    tr.set_defaults(func=train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
