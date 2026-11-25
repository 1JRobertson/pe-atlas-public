"""Voxel features in physical coordinates for single-CT filling-defect proposals."""

import itertools

import numpy as np
from scipy.ndimage import map_coordinates

DIRECTIONS = np.array(
    [v for v in itertools.product((-1, 0, 1), repeat=3) if v > (0, 0, 0)], dtype=float
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]


def features(ct, points, spacing):
    p = points.astype(np.float32)
    center = ct[tuple(points.T)].astype(np.float32)
    cols = [center]
    for radius in (1.5, 3.0, 5.0, 8.0, 13.0):
        endpoints = []
        for direction in DIRECTIONS:
            offset = (radius * direction / spacing).astype(np.float32)
            ends = [
                map_coordinates(
                    ct,
                    (p + sign * offset).T,
                    order=1,
                    mode="nearest",
                    output=np.float32,
                    prefilter=False,
                )
                for sign in (-1, 1)
            ]
            endpoints.append(ends)
        ends = np.array(endpoints)
        low = np.sort(ends.min(axis=1), axis=0)
        high = np.sort(ends.max(axis=1), axis=0)
        all_vals = ends.reshape(-1, len(p))
        cols.extend(
            (
                low[-1] - center,
                low[-3] - center,
                low[-6] - center,
                high[-2] - center,
                high[-5] - center,
                high[6] - center,
                low[4],
                high[1],
                (all_vals > 140).mean(axis=0),
                (all_vals < -350).mean(axis=0),
                (all_vals > center + 55).mean(axis=0),
                np.minimum(400, np.std(all_vals, axis=0)),
            )
        )
    return np.stack(cols, axis=1).astype(np.float32)


def predict(ct, artery, spacing, model):
    from scipy import ndimage as nd

    score = np.zeros(ct.shape, np.uint8)
    # Only consider soft tissue close to the independently segmented arterial lumen.
    for z in range(0, len(ct), 32):
        lo = max(0, z - 4)
        stop = min(len(ct), z + 32)
        hi = min(len(ct), stop + 4)
        a = artery[lo:hi]
        if not np.any(a):
            continue
        margin = nd.distance_transform_edt(~a, sampling=spacing)[z - lo : stop - lo]
        hu = ct[z:stop]
        mask = (margin < 2.3) & (hu >= -35) & (hu <= 205)
        pts = np.argwhere(mask)
        pts[:, 0] += z
        for batch in np.array_split(pts, max(1, int(np.ceil(len(pts) / 30000)))):
            if len(batch):
                prob = model.positive(features(ct, batch, spacing))
                score[tuple(batch.T)] = np.rint(prob * 255).astype(np.uint8)
    return score
