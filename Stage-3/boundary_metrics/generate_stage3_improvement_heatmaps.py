from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Example of paths
FIGURINES_BASE = "/home/scur0806/Appearance-Invariant-Gaussian-Grouping/Stage-3/Gaussian-in-the-Wild/output/figurines_stage3_70k"
RAMEN_BASE = "/home/scur0806/Appearance-Invariant-Gaussian-Grouping/Stage-3/Gaussian-in-the-Wild/output/ramen_stage3_70k"
TEATIME_BASE = "/home/scur0806/Appearance-Invariant-Gaussian-Grouping/Stage-3/Gaussian-in-the-Wild/output/teatime_stage3_70k"

FIGURINES_BASE_IDENTITY = "/scratch-shared/gpuuva074/gaussian-wild/figurines_finetuned_identity_phase_1"
RAMEN_BASE_IDENTITY = "/scratch-shared/gpuuva074/gaussian-wild/ramen_finetuned_identity_phase_1"
TEATIME_BASE_IDENTITY = "/scratch-shared/gpuuva074/gaussian-wild/teatime_finetuned_identity_phase_1"

FIGURINES_BASE_BASELINE = "/scratch-shared/gpuuva074/gaussian-wild/figurines_baseline_phase_1"
RAMEN_BASE_BASELINE = "/scratch-shared/gpuuva074/gaussian-wild/ramen_baseline_phase_1"
TEATIME_BASE_BASELINE = "/scratch-shared/gpuuva074/gaussian-wild/teatime_baseline_phase_1"


BOUNDARY_WIDTH = 5
OUTPUT_ROOT = Path("metric_results") / "stage3_improvement_heatmaps"
PER_IMAGE_LIMIT = None


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
        mask_dir=Path("/scratch-shared/gpuuva074/datasets/figurines/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path(f"{FIGURINES_BASE_BASELINE}/test/ours_70000/renders"), Path(f"{FIGURINES_BASE_BASELINE}/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path(f"{FIGURINES_BASE_IDENTITY}/test/ours_70000/renders"), Path(f"{FIGURINES_BASE_IDENTITY}/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path(f"{FIGURINES_BASE}/test/ours_70000/renders"), Path(f"{FIGURINES_BASE}/test/ours_70000/gt")),
        ],
    ),
    SceneConfig(
        name="ramen",
        mask_dir=Path("/scratch-shared/gpuuva074/datasets/ramen/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path(f"{RAMEN_BASE_BASELINE}/test/ours_70000/renders"), Path(f"{RAMEN_BASE_BASELINE}/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path(f"{RAMEN_BASE_IDENTITY}/test/ours_70000/renders"), Path(f"{RAMEN_BASE_IDENTITY}/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path(f"{RAMEN_BASE}/test/ours_70000/renders"), Path(f"{RAMEN_BASE}/test/ours_70000/gt")),
        ],
    ),
    SceneConfig(
        name="teatime",
        mask_dir=Path("/scratch-shared/gpuuva074/datasets/teatime/boundary_instance"),
        runs=[
            RunConfig("Baseline", Path(f"{TEATIME_BASE_BASELINE}/test/ours_70000/renders"), Path(f"{TEATIME_BASE_BASELINE}/test/ours_70000/gt")),
            RunConfig("Finetuned identity", Path(f"{TEATIME_BASE_IDENTITY}/test/ours_70000/renders"), Path(f"{TEATIME_BASE_IDENTITY}/test/ours_70000/gt")),
            RunConfig("Stage 3 boundary", Path(f"{TEATIME_BASE}/test/ours_70000/renders"), Path(f"{TEATIME_BASE}/test/ours_70000/gt")),
        ],
    ),
]


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


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
        candidate_stems.append(f"frame_{stem}")

    seen = set()
    unique_stems = []
    for candidate in candidate_stems:
        if candidate not in seen:
            seen.add(candidate)
            unique_stems.append(candidate)

    for candidate in unique_stems:
        for ext in [".npy", ".png", ".jpg", ".jpeg"]:
            path = mask_dir / f"{candidate}{ext}"
            if path.exists():
                return path
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


def pixel_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - gt), axis=2)


def save_heatmap(values: np.ndarray, out_path: Path, title: str, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Improvement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_rg_overlay(base_rgb: np.ndarray, values: np.ndarray, mask: np.ndarray, vmax: float, max_alpha: float = 0.7) -> np.ndarray:
    norm = np.clip(np.abs(values) / max(vmax, 1e-8), 0.0, 1.0)
    alpha = norm * max_alpha
    alpha = np.where(mask, alpha, 0.0)

    overlay = np.zeros_like(base_rgb)
    positive = values > 0
    negative = values < 0

    # Positive improvement -> red, negative improvement -> green.
    overlay[..., 0] = np.where(positive, 1.0, 0.0)
    overlay[..., 1] = np.where(negative, 1.0, 0.0)

    blended = base_rgb * (1.0 - alpha[..., None]) + overlay * alpha[..., None]
    return np.clip(blended, 0.0, 1.0)


def save_overlay(base_rgb: np.ndarray, values: np.ndarray, mask: np.ndarray, out_path: Path, title: str, vmax: float) -> None:
    blended = make_rg_overlay(base_rgb, values, mask, vmax)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(blended)
    ax.set_title(title)
    ax.axis("off")

    # Standalone colorbar for the improvement scale.
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Improvement")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def ensure_matching_shape(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == image_hw:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.int32)).resize((image_hw[1], image_hw[0]), Image.NEAREST))


def scene_run_map(scene: SceneConfig) -> dict[str, RunConfig]:
    return {run.label: run for run in scene.runs}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        runs = scene_run_map(scene)
        stage3 = runs["Stage 3 boundary"]
        comparisons = [
            ("vs_baseline", runs["Baseline"]),
            ("vs_identity", runs["Finetuned identity"]),
        ]

        scene_dir = OUTPUT_ROOT / scene.name
        scene_dir.mkdir(parents=True, exist_ok=True)

        stage3_render_paths = sorted(stage3.renders_dir.glob("*.png"))
        if PER_IMAGE_LIMIT is not None:
            stage3_render_paths = stage3_render_paths[:PER_IMAGE_LIMIT]

        full_accums = {name: [] for name, _ in comparisons}
        boundary_accums = {name: [] for name, _ in comparisons}
        gt_accums = []

        for render_path in stage3_render_paths:
            stem = render_path.stem
            gt_path = stage3.gt_dir / render_path.name
            mask_path = find_mask(scene.mask_dir, stem)

            stage3_img = load_image(render_path)
            gt_img = load_image(gt_path)
            gt_accums.append(gt_img)
            mask = ensure_matching_shape(load_mask(mask_path), stage3_img.shape[:2])
            band = compute_boundary_band(mask, BOUNDARY_WIDTH)

            stage3_err = pixel_error_map(stage3_img, gt_img)

            for comp_name, reference_run in comparisons:
                ref_img = load_image(reference_run.renders_dir / render_path.name)
                ref_err = pixel_error_map(ref_img, gt_img)
                improvement = ref_err - stage3_err
                boundary_improvement = np.where(band, improvement, 0.0)

                full_accums[comp_name].append(improvement)
                boundary_accums[comp_name].append(boundary_improvement)

                vmax = max(np.max(np.abs(improvement)), 1e-6)
                per_image_dir = scene_dir / comp_name / "per_image"
                per_image_dir.mkdir(parents=True, exist_ok=True)

                save_heatmap(
                    improvement,
                    per_image_dir / f"{stem}_full.png",
                    f"{scene.name} {comp_name} full {stem}",
                    vmax,
                )
                save_overlay(
                    gt_img,
                    improvement,
                    np.ones_like(band, dtype=bool),
                    per_image_dir / f"{stem}_full_overlay.png",
                    f"{scene.name} {comp_name} full overlay {stem}",
                    vmax,
                )
                save_overlay(
                    gt_img,
                    improvement,
                    band,
                    per_image_dir / f"{stem}_boundary_overlay.png",
                    f"{scene.name} {comp_name} boundary overlay {stem}",
                    vmax,
                )

        for comp_name, _reference_run in comparisons:
            if not full_accums[comp_name]:
                continue

            full_mean = np.mean(np.stack(full_accums[comp_name], axis=0), axis=0)
            boundary_mean = np.mean(np.stack(boundary_accums[comp_name], axis=0), axis=0)
            mean_gt = np.mean(np.stack(gt_accums, axis=0), axis=0)
            vmax = max(np.max(np.abs(full_mean)), np.max(np.abs(boundary_mean)), 1e-6)

            avg_dir = scene_dir / comp_name
            save_heatmap(
                full_mean,
                avg_dir / "average_full.png",
                f"{scene.name} {comp_name} average full",
                vmax,
            )
            save_overlay(
                mean_gt,
                full_mean,
                np.ones_like(full_mean, dtype=bool),
                avg_dir / "average_full_overlay.png",
                f"{scene.name} {comp_name} average full overlay",
                vmax,
            )
            save_overlay(
                mean_gt,
                boundary_mean,
                boundary_mean != 0.0,
                avg_dir / "average_boundary_overlay.png",
                f"{scene.name} {comp_name} average boundary overlay",
                vmax,
            )

            np.save(avg_dir / "average_full.npy", full_mean)
            np.save(avg_dir / "average_boundary.npy", boundary_mean)

        print(f"Finished {scene.name}")


if __name__ == "__main__":
    main()
