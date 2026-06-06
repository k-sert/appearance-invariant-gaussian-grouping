import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

try:
    import lpips
except ImportError as exc:
    raise SystemExit("Please install lpips in the active environment: pip install lpips") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOUNDARY_WIDTH = 5
OUTPUT_DIR = Path("metric_results") / "boundary_band_metrics"


@dataclass
class RunConfig:
    label: str
    renders_dir: Path
    gt_dir: Path


@dataclass
class SceneConfig:
    name: str
    mask_dir: Path
    runs: list[RunConfig]

# Example of paths
SCENES = [
    SceneConfig(
        name="figurines",
        mask_dir=Path("/path/to/figurines/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path("/path/to/figurines_baseline/test/ours_70000/renders"), Path("/path/to/figurines_baseline/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path("/path/to/figurines_identity/test/ours_70000/renders"), Path("/path/to/figurines_identity/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path("/path/to/figurines_stage3_70k/test/ours_70000/renders"), Path("/path/to/figurines_stage3_70k/test/ours_70000/gt")),
        ],
    ),
    SceneConfig(
        name="ramen",
        mask_dir=Path("/path/to/ramen/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path("/path/to/ramen_baseline/test/ours_70000/renders"), Path("/path/to/ramen_baseline/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path("/path/to/ramen_identity/test/ours_70000/renders"), Path("/path/to/ramen_identity/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path("/path/to/ramen_stage3_70k/test/ours_70000/renders"), Path("/path/to/ramen_stage3_70k/test/ours_70000/gt")),
        ],
    ),
    SceneConfig(
        name="teatime",
        mask_dir=Path("/path/to/teatime/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path("/path/to/teatime_baseline/test/ours_70000/renders"), Path("/path/to/teatime_baseline/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path("/path/to/teatime_identity/test/ours_70000/renders"), Path("/path/to/teatime_identity/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path("/path/to/teatime_stage3_70k/test/ours_70000/renders"), Path("/path/to/teatime_stage3_70k/gt")),
        ],
    ),
]


def load_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path))

    if mask.ndim == 3:
        flat = mask.reshape(-1, mask.shape[-1])
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        mask = inv.reshape(mask.shape[:2])

    return mask.astype(np.int64)


def find_mask(mask_dir: Path, stem: str) -> Path:
    candidate_stems = [stem]
    try:
        frame_idx = int(stem)
        candidate_stems.extend(
            [
                f"{frame_idx:05d}",
                f"{frame_idx + 1:05d}",
                f"frame_{frame_idx:05d}",
                f"frame_{frame_idx + 1:05d}",
            ]
        )
    except ValueError:
        candidate_stems.extend([f"frame_{stem}"])

    seen = set()
    unique_candidate_stems = []
    for candidate_stem in candidate_stems:
        if candidate_stem not in seen:
            seen.add(candidate_stem)
            unique_candidate_stems.append(candidate_stem)

    for candidate_stem in unique_candidate_stems:
        for ext in [".npy", ".png", ".jpg", ".jpeg"]:
            candidate = mask_dir / f"{candidate_stem}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No mask found for {stem} in {mask_dir}")


def compute_boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]

    if width <= 1:
        return edge

    band = edge.copy()
    for dy in range(-width + 1, width):
        for dx in range(-width + 1, width):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(edge)
            y_src_start = max(0, -dy)
            y_src_end = min(edge.shape[0], edge.shape[0] - dy)
            x_src_start = max(0, -dx)
            x_src_end = min(edge.shape[1], edge.shape[1] - dx)
            y_dst_start = max(0, dy)
            y_dst_end = min(edge.shape[0], edge.shape[0] + dy)
            x_dst_start = max(0, dx)
            x_dst_end = min(edge.shape[1], edge.shape[1] + dx)
            shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = edge[y_src_start:y_src_end, x_src_start:x_src_end]
            band |= shifted
    return band


def masked_psnr(pred: np.ndarray, gt: np.ndarray, band: np.ndarray) -> float:
    diff = pred[band] - gt[band]
    mse = np.mean(diff ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def masked_ssim(pred: np.ndarray, gt: np.ndarray, band: np.ndarray) -> float:
    _, ssim_map = structural_similarity(pred, gt, channel_axis=-1, data_range=1.0, full=True)
    return float(ssim_map[band].mean())


def masked_lpips(lpips_model, pred: np.ndarray, gt: np.ndarray, band: np.ndarray) -> float:
    ys, xs = np.where(band)
    if ys.size == 0:
        return float("nan")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop_mask = band[y0:y1, x0:x1].astype(np.float32)[..., None]
    pred_crop = pred[y0:y1, x0:x1] * crop_mask
    gt_crop = gt[y0:y1, x0:x1] * crop_mask

    pred_t = torch.from_numpy(pred_crop).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    gt_t = torch.from_numpy(gt_crop).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    with torch.no_grad():
        value = lpips_model(pred_t, gt_t).item()
    return float(value)


def evaluate_run(scene: SceneConfig, run: RunConfig, lpips_model) -> dict:
    render_paths = sorted(run.renders_dir.glob("*.png"))
    if not render_paths:
        raise FileNotFoundError(f"No renders found in {run.renders_dir}")

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for render_path in render_paths:
        gt_path = run.gt_dir / render_path.name
        mask_path = find_mask(scene.mask_dir, render_path.stem)

        pred = load_image(render_path)
        gt = load_image(gt_path)
        mask = load_mask(mask_path)

        if mask.shape != pred.shape[:2]:
            mask = np.asarray(Image.fromarray(mask.astype(np.int32)).resize((pred.shape[1], pred.shape[0]), Image.NEAREST))

        band = compute_boundary_band(mask, BOUNDARY_WIDTH)
        if band.sum() == 0:
            continue

        psnr_values.append(masked_psnr(pred, gt, band))
        ssim_values.append(masked_ssim(pred, gt, band))
        lpips_values.append(masked_lpips(lpips_model, pred, gt, band))

    return {
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
        "num_images": len(psnr_values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute boundary-band PSNR/SSIM/LPIPS for configured scenes or a single run."
    )
    parser.add_argument(
        "--single-run-root",
        type=Path,
        default=None,
        help=(
            "Optional path to a single experiment root, e.g. "
            ".../Gaussian-in-the-Wild/output/figurines_stage3_70k_diff_bound_loss_coeff. "
            "When set, only this run is evaluated."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Mask directory to use with --single-run-root, e.g. .../datasets/figurines/boundary_instance.",
    )
    parser.add_argument(
        "--scene-name",
        default="single_run",
        help="Scene name label to use in output when --single-run-root is set.",
    )
    parser.add_argument(
        "--run-label",
        default="Stage 3 boundary",
        help="Run label to use in output when --single-run-root is set.",
    )
    return parser


def build_single_scene(single_run_root: Path, mask_dir: Path, scene_name: str, run_label: str) -> SceneConfig:
    run_root = single_run_root.expanduser().resolve()
    mask_dir = mask_dir.expanduser().resolve()
    renders_dir = run_root / "test" / "ours_70000" / "renders"
    gt_dir = run_root / "test" / "ours_70000" / "gt"

    if not renders_dir.exists():
        raise FileNotFoundError(f"Renders directory does not exist: {renders_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    return SceneConfig(
        name=scene_name,
        mask_dir=mask_dir,
        runs=[RunConfig(run_label, renders_dir, gt_dir)],
    )


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model.eval()

    summary = {}
    rows = []

    scenes = SCENES
    if args.single_run_root is not None:
        if args.mask_dir is None:
            raise SystemExit("--mask-dir is required when using --single-run-root")
        scenes = [
            build_single_scene(
                single_run_root=args.single_run_root,
                mask_dir=args.mask_dir,
                scene_name=args.scene_name,
                run_label=args.run_label,
            )
        ]

    for scene in scenes:
        summary[scene.name] = {}
        for run in scene.runs:
            metrics = evaluate_run(scene, run, lpips_model)
            summary[scene.name][run.label] = metrics
            rows.append(
                {
                    "scene": scene.name,
                    "run": run.label,
                    "boundary_width": BOUNDARY_WIDTH,
                    **metrics,
                }
            )
            print(f"{scene.name:10s} | {run.label:18s} | "
                  f"PSNR={metrics['PSNR']:.6f} SSIM={metrics['SSIM']:.6f} "
                  f"LPIPS={metrics['LPIPS']:.6f} N={metrics['num_images']}")

    json_path = OUTPUT_DIR / "boundary_band_metrics.json"
    csv_path = OUTPUT_DIR / "boundary_band_metrics.csv"

    json_path.write_text(json.dumps(summary, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "run", "boundary_width", "PSNR", "SSIM", "LPIPS", "num_images"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON: {json_path.resolve()}")
    print(f"Saved CSV:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
