"""Grade low-attenuation proposals using surrounding contrast on one native CT."""

from itertools import combinations

import numpy as np
from appearance import DIRECTIONS
from scipy import ndimage as nd

CONNECTED = np.ones((3, 3, 3), bool)


def contrast_core(ct, artery, region, spacing):
    """Return candidate voxels bracketed by blood with no short dark exit into soft tissue."""
    points = np.argwhere(region)
    index = tuple(points.T)
    depth = nd.distance_transform_edt(artery, sampling=spacing)
    smooth = nd.gaussian_filter(ct.astype(np.float32), 0.48 / spacing)
    nearby = nd.distance_transform_edt(~region, sampling=spacing) < 8
    blood = smooth[nearby & artery & (smooth >= 60)]
    core = np.zeros_like(region)
    if not len(blood):
        return core
    local = np.clip(np.percentile(blood, 78), 120, 550)
    center = smooth[index]
    threshold = np.maximum(max(107, 0.68 * local), center + max(42, 0.24 * local))
    signed_distance = depth - nd.distance_transform_edt(~artery, sampling=spacing)
    eligible = depth[index] >= 1
    points = points.astype(np.float32)
    votes = []
    soft_exit = np.zeros(len(points), bool)
    for direction in DIRECTIONS:
        opposed = eligible.copy()
        for sign in (-1, 1):
            inside = eligible.copy()
            dark = eligible.copy()
            enhanced = np.zeros(len(points), bool)
            step = (sign * direction * 0.85 / spacing).astype(np.float32)
            for step_number in range(1, 13):
                coords = (points + step_number * step).T
                wall = nd.map_coordinates(signed_distance, coords, order=1, cval=-20)
                hu = nd.map_coordinates(smooth, coords, order=1, cval=-1000)
                inside &= (wall >= 0) & (hu > -40)
                enhanced |= inside & (hu >= threshold)
                if step_number <= 7:
                    dark &= (hu >= -45) & (hu < threshold)
                    soft_exit |= dark & (wall <= -1.5)
            opposed &= enhanced
        votes.append(opposed)
    supported = np.zeros(len(points), bool)
    for i, j in combinations(range(len(DIRECTIONS)), 2):
        if abs(DIRECTIONS[i] @ DIRECTIONS[j]) < 0.76:
            supported |= votes[i] & votes[j]
    core[index] = supported & ~soft_exit
    return core


def grade(ct, artery, proposals, regions, spacing):
    """Retain proposal IDs; 1 is supported extent, 2 is the remaining proposal."""
    labels, _ = nd.label(proposals, CONNECTED)
    boxes = nd.find_objects(labels)
    tiers = proposals.astype(np.uint8) * 2
    margin = np.ceil(13 / spacing).astype(int)
    voxel_mm3 = np.prod(spacing)
    results = []
    for record in regions:
        center = tuple(record["centerIJK"][::-1])
        ident = labels[center]
        if not ident:
            raise ValueError("Proposal center does not map to its source component")
        box = tuple(
            slice(max(0, part.start - pad), min(size, part.stop + pad))
            for part, pad, size in zip(boxes[ident - 1], margin, ct.shape)
        )
        region = labels[box] == ident
        vessel = artery[box]
        core = contrast_core(ct[box], vessel, region, spacing)
        islands, _ = nd.label(core, CONNECTED)
        anchors = np.zeros_like(region)
        for n, slices in enumerate(nd.find_objects(islands), 1):
            blob = islands[slices] == n
            if slices[0].stop - slices[0].start >= 2 and blob.sum() * voxel_mm3 >= 6:
                anchors[slices] |= blob
        red = np.zeros_like(region)
        if (
            anchors.any()
            and (vessel & region).sum() >= 0.5 * region.sum()
            and core.sum() >= 0.01 * region.sum()
        ):
            red = (
                region
                & vessel
                & (nd.distance_transform_edt(~anchors, sampling=spacing) <= 3)
            )
            # A nearby, disconnected piece has no supported path to this anchor.
            parts, _ = nd.label(red, CONNECTED)
            red = np.isin(parts, np.unique(parts[anchors]))
            tiers[box][red] = 1
        result = dict(
            record,
            supportedVoxels=int(red.sum()),
            evidence="contrast-supported" if red.any() else "indeterminate",
        )
        result.pop("evidenceIJK", None)
        if red.any():
            q = np.argwhere(anchors) + [s.start for s in box]
            nearest = q[np.argmin(np.linalg.norm((q - center) * spacing, axis=1))]
            result["evidenceIJK"] = nearest[::-1].tolist()
        results.append(result)
    return tiers, results
