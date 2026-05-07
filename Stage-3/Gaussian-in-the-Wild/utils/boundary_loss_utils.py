import torch
import torch.nn.functional as F


def project_points_to_pixels(xyz, viewpoint_camera):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)
    clip = xyz_h @ viewpoint_camera.full_proj_transform
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)
    x = (ndc[:, 0] * 0.5 + 0.5) * (viewpoint_camera.image_width - 1)
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (viewpoint_camera.image_height - 1)
    return torch.stack([x, y, ndc[:, 2]], dim=1)


def _sample_nearest_map(map_hw, xy):
    h, w = map_hw.shape[-2:]
    x = xy[:, 0].round().long().clamp(0, w - 1)
    y = xy[:, 1].round().long().clamp(0, h - 1)
    return map_hw[y, x]


def _collect_boundary_points(gaussians, viewpoint_camera, visibility_filter, edge_threshold, max_points=None):
    if not hasattr(viewpoint_camera, "boundary_confidence"):
        return None
    if viewpoint_camera.boundary_confidence is None or viewpoint_camera.instance_mask is None:
        return None

    xyz = gaussians.get_xyz
    xy_depth = project_points_to_pixels(xyz, viewpoint_camera)
    xy = xy_depth[:, :2]
    in_frame = (
        (xy[:, 0] >= 0) & (xy[:, 0] < viewpoint_camera.image_width) &
        (xy[:, 1] >= 0) & (xy[:, 1] < viewpoint_camera.image_height) &
        (xy_depth[:, 2] > 0)
    )
    idx = torch.nonzero(visibility_filter & in_frame, as_tuple=False).squeeze(-1)
    if idx.numel() < 2:
        return None

    edge = viewpoint_camera.boundary_confidence.to(xyz.device).float()
    labels = viewpoint_camera.instance_mask.to(xyz.device).long()
    edge_values = _sample_nearest_map(edge, xy[idx])
    keep = edge_values > edge_threshold
    idx = idx[keep]
    edge_values = edge_values[keep]
    if idx.numel() < 2:
        return None

    if max_points is not None and idx.numel() > max_points:
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
        idx = idx[perm]
        edge_values = edge_values[perm]

    xy_sel = xy[idx]
    labels_sel = _sample_nearest_map(labels, xy_sel)
    return idx, xy_sel, labels_sel, edge_values


def boundary_identity_contrast_loss(
    gaussians,
    viewpoint_camera,
    visibility_filter,
    max_points=4096,
    k_neighbors=8,
    edge_threshold=0.2,
    similarity_margin=0.25,
):
    if not hasattr(gaussians, "_identity") or gaussians._identity.numel() == 0:
        return gaussians.get_xyz.sum() * 0.0

    gathered = _collect_boundary_points(
        gaussians, viewpoint_camera, visibility_filter, edge_threshold, max_points=max_points
    )
    if gathered is None:
        return gaussians.get_xyz.sum() * 0.0

    idx, xy_sel, labels_sel, edge_values = gathered
    identity = gaussians._identity[idx]
    if not getattr(gaussians, "boundary_identity_trainable", False):
        identity = identity.detach()
    ids = F.normalize(identity, dim=1, eps=1e-6)

    k = min(k_neighbors + 1, idx.numel())
    scale = max(viewpoint_camera.image_width, viewpoint_camera.image_height)
    dist = torch.cdist(xy_sel / scale, xy_sel / scale)
    nn = torch.topk(dist, k=k, largest=False).indices[:, 1:]

    different_instance = labels_sel[:, None] != labels_sel[nn]
    if not different_instance.any():
        return gaussians.get_xyz.sum() * 0.0

    sim = (ids[:, None, :] * ids[nn]).sum(dim=-1)
    weights = torch.minimum(edge_values[:, None], edge_values[nn])
    penalty = F.relu(sim - similarity_margin) * weights
    return penalty[different_instance].mean()


def boundary_footprint_loss(
    gaussians,
    viewpoint_camera,
    visibility_filter,
    radii,
    max_points=2048,
    edge_threshold=0.2,
    overlap_margin=1.0,
):
    gathered = _collect_boundary_points(
        gaussians, viewpoint_camera, visibility_filter, edge_threshold, max_points=max_points
    )
    if gathered is None:
        return gaussians.get_xyz.sum() * 0.0

    idx, xy_sel, labels_sel, edge_values = gathered
    if idx.numel() < 2:
        return gaussians.get_xyz.sum() * 0.0

    radii_sel = radii[idx].float().clamp_min(0.0)
    valid_radius = radii_sel > 0
    if valid_radius.sum() < 2:
        return gaussians.get_xyz.sum() * 0.0

    xy_sel = xy_sel[valid_radius]
    labels_sel = labels_sel[valid_radius]
    edge_values = edge_values[valid_radius]
    radii_sel = radii_sel[valid_radius]

    pair_dist = torch.cdist(xy_sel, xy_sel)
    combined_radii = overlap_margin * (radii_sel[:, None] + radii_sel[None, :])
    different_instance = labels_sel[:, None] != labels_sel[None, :]
    overlap = F.relu(combined_radii - pair_dist) / (combined_radii + 1e-6)
    weights = torch.minimum(edge_values[:, None], edge_values[None, :])
    penalty = overlap * weights
    penalty = penalty[different_instance]
    if penalty.numel() == 0:
        return gaussians.get_xyz.sum() * 0.0
    return penalty.mean()
