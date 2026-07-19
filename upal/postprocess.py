"""Small inference helpers for matching descriptors and extracting line segments."""

from __future__ import annotations

import numpy as np
import torch


def mutual_nearest_neighbors(
    descriptors0: torch.Tensor,
    descriptors1: torch.Tensor,
) -> torch.Tensor:
    """Return mutual descriptor matches as an ``M x 2`` index tensor."""
    similarity = descriptors0 @ descriptors1.T
    nearest1 = similarity.argmax(dim=1)
    nearest0 = similarity.argmax(dim=0)
    index0 = torch.arange(len(descriptors0), device=descriptors0.device)
    mutual = nearest0[nearest1] == index0
    selected = index0[mutual]
    return torch.stack([selected, nearest1[selected]], dim=1)


def detect_lines(
    image: torch.Tensor,
    distance_field: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    max_lines: int = 200,
    min_length: float = 25.0,
    max_mean_distance: float = 2.0,
) -> np.ndarray:
    """Detect point-seeded LSD segments and retain line-field-supported proposals.

    Gradients, keypoint seeds, and learned-field filtering run in Torch on the
    input device. The detector is the ``pytlsd`` extension built from the
    bundled ``points_lsd`` submodule; its NumPy C++ API is the sole CPU stage.
    """
    try:
        import pytlsd
    except ImportError as error:
        raise ImportError(
            "points_lsd is required for line detection; run "
            "`git submodule update --init --recursive` followed by "
            "`DEBUG=0 python3 -m pip install ./third_party/points_lsd`."
        ) from error

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("line detection accepts one image at a time")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("image must have shape 1xHxW or 3xHxW")
    if distance_field.ndim != 2:
        raise ValueError("distance_field must have shape HxW")

    if image.shape[0] == 3:
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        gray = (image * weights).sum(dim=0)
    else:
        gray = image[0]
    gray = gray.mul(255.0)
    gradients, angles = _lsd_gradients(gray)
    seeds = _line_seed_points(keypoints, gray.shape)
    if len(seeds) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)

    segments = pytlsd.lsd_from_points(
        np.ascontiguousarray(gray.detach().cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(seeds.detach().cpu().numpy(), dtype=np.int32),
        1.0,
        0.6,
        0.0,
        np.ascontiguousarray(gradients.detach().cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(angles.detach().cpu().numpy(), dtype=np.float64),
    )
    if len(segments) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)

    lines = torch.from_numpy(segments[:, :4]).to(distance_field).reshape(-1, 2, 2)
    line_scores = torch.from_numpy(segments[:, 4]).to(distance_field)
    lengths = torch.linalg.vector_norm(lines[:, 1] - lines[:, 0], dim=1)
    keep = lengths >= min_length
    lines, line_scores, lengths = lines[keep], line_scores[keep], lengths[keep]
    if len(lines) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)

    samples = lines[:, :1] + (lines[:, 1:] - lines[:, :1]) * torch.linspace(
        0, 1, 32, device=lines.device
    ).view(1, -1, 1)
    x = samples[..., 0].round().long().clamp_(0, distance_field.shape[1] - 1)
    y = samples[..., 1].round().long().clamp_(0, distance_field.shape[0] - 1)
    mean_distance = distance_field[y, x].mean(dim=1)
    keep = mean_distance <= max_mean_distance
    lines, line_scores, lengths, mean_distance = (
        lines[keep],
        line_scores[keep],
        lengths[keep],
        mean_distance[keep],
    )
    order = torch.argsort(-lengths, stable=True)
    order = order[torch.argsort(-line_scores[order], stable=True)]
    order = order[torch.argsort(mean_distance[order], stable=True)]
    return lines[order[:max_lines]].detach().cpu().numpy()


def _line_seed_points(keypoints: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    """Convert predicted ``(x, y)`` points to valid, unique LSD seeds."""
    height, width = image_shape
    points = keypoints.round().to(dtype=torch.int32).reshape(-1, 2)
    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return torch.unique(points[valid], dim=0).contiguous()


def _lsd_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute point-LSD gradients in Torch on the image's device."""
    not_defined = -1024.0
    gradient_norm = torch.full_like(image, not_defined)
    gradient_angle = torch.full_like(image, not_defined)
    a, b = image[:-1, :-1], image[:-1, 1:]
    c, d = image[1:, :-1], image[1:, 1:]
    horizontal = b + d - a - c
    vertical = c + d - a - b
    gradient_norm[:-1, :-1] = 0.5 * torch.hypot(horizontal, vertical)
    gradient_angle[:-1, :-1] = torch.atan2(horizontal, -vertical)
    gradient_angle.masked_fill_(gradient_norm <= 5.2262518595055063, not_defined)
    return gradient_norm, gradient_angle


def match_lines_from_endpoints(
    endpoint_descriptors0: torch.Tensor,
    endpoint_descriptors1: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Globally match lines from their endpoint descriptors.

    The score for a pair is the best average descriptor similarity over the two
    possible endpoint orientations. A maximum-weight one-to-one assignment then
    selects the line matches.
    """
    empty_matches = np.empty((0, 2), dtype=np.int64)
    empty_scores = np.empty((0,), dtype=np.float32)
    if endpoint_descriptors0.ndim != 3 or endpoint_descriptors1.ndim != 3:
        raise ValueError("endpoint descriptors must have shape L x 2 x D")
    if endpoint_descriptors0.shape[1:] != endpoint_descriptors1.shape[1:] or endpoint_descriptors0.shape[1] != 2:
        raise ValueError("endpoint descriptors must have matching shape L x 2 x D")
    if len(endpoint_descriptors0) == 0 or len(endpoint_descriptors1) == 0:
        return empty_matches, empty_scores

    similarities = torch.einsum("iad,jbd->iajb", endpoint_descriptors0, endpoint_descriptors1)
    aligned = similarities[:, 0, :, 0] + similarities[:, 1, :, 1]
    reversed_ = similarities[:, 0, :, 1] + similarities[:, 1, :, 0]
    scores = (0.5 * torch.maximum(aligned, reversed_)).detach().cpu().numpy()
    if scores.shape[0] <= scores.shape[1]:
        index0 = np.arange(scores.shape[0])
        index1 = _maximum_weight_assignment(scores)
    else:
        index1 = np.arange(scores.shape[1])
        index0 = _maximum_weight_assignment(scores.T)
    matches = np.column_stack((index0, index1)).astype(np.int64)
    match_scores = scores[index0, index1].astype(np.float32)
    order = np.argsort(-match_scores)
    return matches[order], match_scores[order]


def _maximum_weight_assignment(scores: np.ndarray) -> np.ndarray:
    """Return the maximum-weight assignment for a matrix with rows <= columns."""
    rows, columns = scores.shape
    if rows > columns:
        raise ValueError("assignment requires at least as many columns as rows")
    costs = -scores.astype(np.float64, copy=False)
    potentials_rows = np.zeros(rows + 1)
    potentials_columns = np.zeros(columns + 1)
    matching = np.zeros(columns + 1, dtype=np.int64)
    predecessor = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        matching[0] = row
        column0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = matching[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                value = costs[row0 - 1, column - 1] - potentials_rows[row0] - potentials_columns[column]
                if value < minimum[column]:
                    minimum[column] = value
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta, column1 = minimum[column], column
            for column in range(columns + 1):
                if used[column]:
                    potentials_rows[matching[column]] += delta
                    potentials_columns[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matching[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matching[column0] = matching[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.empty(rows, dtype=np.int64)
    for column in range(1, columns + 1):
        if matching[column]:
            assignment[matching[column] - 1] = column - 1
    return assignment
