import torch
import torch.nn.functional as F


def _empty_loss(rendered_identity: torch.Tensor) -> torch.Tensor:
    return rendered_identity.sum() * 0.0


def _prepare_boundary_maps(rendered_identity, viewpoint_camera, edge_threshold):
    if rendered_identity is None:
        return None
    if not hasattr(viewpoint_camera, "boundary_confidence"):
        return None
    if viewpoint_camera.boundary_confidence is None or viewpoint_camera.instance_mask is None:
        return None

    feature = rendered_identity.permute(1, 2, 0).contiguous()
    feat_norm = F.normalize(feature, dim=-1, eps=1e-6)

    edge = viewpoint_camera.boundary_confidence.to(feature.device).float()
    labels = viewpoint_camera.instance_mask.to(feature.device).long()
    valid = torch.linalg.norm(feature, dim=-1) > 1e-6
    edge_mask = (edge > edge_threshold) & valid
    if edge_mask.sum() == 0:
        return None
    return feat_norm, labels, edge, edge_mask, valid


def _subsample_pairs(mask, max_pairs):
    if max_pairs is None:
        return mask
    count = int(mask.sum().item())
    if count <= max_pairs:
        return mask

    keep = torch.zeros(count, dtype=torch.bool, device=mask.device)
    perm = torch.randperm(count, device=mask.device)[:max_pairs]
    keep[perm] = True

    out = torch.zeros_like(mask)
    out[mask] = keep
    return out


def boundary_identity_contrast_loss(
    rendered_identity,
    viewpoint_camera,
    max_points=4096,
    k_neighbors=8,
    edge_threshold=0.2,
    similarity_margin=0.25,
):
    del k_neighbors
    prepared = _prepare_boundary_maps(rendered_identity, viewpoint_camera, edge_threshold)
    if prepared is None:
        return _empty_loss(rendered_identity)

    feat_norm, labels, edge, edge_mask, valid = prepared

    horiz_mask = edge_mask[:, :-1] | edge_mask[:, 1:]
    horiz_mask &= valid[:, :-1] & valid[:, 1:]
    horiz_mask &= labels[:, :-1] != labels[:, 1:]

    vert_mask = edge_mask[:-1, :] | edge_mask[1:, :]
    vert_mask &= valid[:-1, :] & valid[1:, :]
    vert_mask &= labels[:-1, :] != labels[1:, :]

    horiz_mask = _subsample_pairs(horiz_mask, max_points // 2 if max_points is not None else None)
    vert_mask = _subsample_pairs(vert_mask, max_points // 2 if max_points is not None else None)

    losses = []
    if horiz_mask.any():
        sim_h = (feat_norm[:, :-1, :] * feat_norm[:, 1:, :]).sum(dim=-1)
        weight_h = torch.minimum(edge[:, :-1], edge[:, 1:])
        losses.append(F.relu(sim_h - similarity_margin)[horiz_mask] * weight_h[horiz_mask])
    if vert_mask.any():
        sim_v = (feat_norm[:-1, :, :] * feat_norm[1:, :, :]).sum(dim=-1)
        weight_v = torch.minimum(edge[:-1, :], edge[1:, :])
        losses.append(F.relu(sim_v - similarity_margin)[vert_mask] * weight_v[vert_mask])

    if not losses:
        return _empty_loss(rendered_identity)
    return torch.cat(losses).mean()


def boundary_footprint_loss(
    rendered_identity,
    viewpoint_camera,
    max_points=2048,
    edge_threshold=0.2,
    overlap_margin=0.1,
):
    prepared = _prepare_boundary_maps(rendered_identity, viewpoint_camera, edge_threshold)
    if prepared is None:
        return _empty_loss(rendered_identity)

    feat_norm, labels, edge, edge_mask, valid = prepared
    losses = []

    # Horizontal boundary: compare each side to its own interior neighbor,
    # then penalize when the cross-edge identity is more similar.
    if feat_norm.shape[1] >= 4:
        pair_mask = (edge_mask[:, 1:-2] | edge_mask[:, 2:-1])
        pair_mask &= valid[:, 1:-2] & valid[:, 2:-1]
        pair_mask &= labels[:, 1:-2] != labels[:, 2:-1]
        pair_mask &= valid[:, :-3] & (labels[:, :-3] == labels[:, 1:-2])
        pair_mask &= valid[:, 3:] & (labels[:, 3:] == labels[:, 2:-1])

        if pair_mask.any():
            left_anchor = feat_norm[:, 1:-2, :]
            right_anchor = feat_norm[:, 2:-1, :]
            left_pos_feat = feat_norm[:, :-3, :]
            right_pos_feat = feat_norm[:, 3:, :]
            cross_sim = (left_anchor * right_anchor).sum(dim=-1)
            left_pos_sim = (left_anchor * left_pos_feat).sum(dim=-1)
            right_pos_sim = (right_anchor * right_pos_feat).sum(dim=-1)
            weight = torch.minimum(edge[:, 1:-2], edge[:, 2:-1])

            pair_mask = _subsample_pairs(pair_mask, max_points // 2 if max_points is not None else None)
            losses.append(F.relu(cross_sim - left_pos_sim + overlap_margin)[pair_mask] * weight[pair_mask])
            losses.append(F.relu(cross_sim - right_pos_sim + overlap_margin)[pair_mask] * weight[pair_mask])

    # Vertical boundary equivalent.
    if feat_norm.shape[0] >= 4:
        pair_mask = (edge_mask[1:-2, :] | edge_mask[2:-1, :])
        pair_mask &= valid[1:-2, :] & valid[2:-1, :]
        pair_mask &= labels[1:-2, :] != labels[2:-1, :]
        pair_mask &= valid[:-3, :] & (labels[:-3, :] == labels[1:-2, :])
        pair_mask &= valid[3:, :] & (labels[3:, :] == labels[2:-1, :])
        if pair_mask.any():
            top_anchor = feat_norm[1:-2, :, :]
            bottom_anchor = feat_norm[2:-1, :, :]
            top_pos_feat = feat_norm[:-3, :, :]
            bottom_pos_feat = feat_norm[3:, :, :]
            cross_sim = (top_anchor * bottom_anchor).sum(dim=-1)
            top_pos_sim = (top_anchor * top_pos_feat).sum(dim=-1)
            bottom_pos_sim = (bottom_anchor * bottom_pos_feat).sum(dim=-1)
            weight = torch.minimum(edge[1:-2, :], edge[2:-1, :])

            pair_mask = _subsample_pairs(pair_mask, max_points // 2 if max_points is not None else None)
            losses.append(F.relu(cross_sim - top_pos_sim + overlap_margin)[pair_mask] * weight[pair_mask])
            losses.append(F.relu(cross_sim - bottom_pos_sim + overlap_margin)[pair_mask] * weight[pair_mask])

    if not losses:
        return _empty_loss(rendered_identity)
    return torch.cat(losses).mean()
