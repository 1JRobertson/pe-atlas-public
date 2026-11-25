"""Study-bound artery guides and user pins in the original DICOM coordinate system."""

import fcntl
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np


def course_frames(points, previous=None):
    """Return a centered plane across the local course at each scroll position."""
    segments = np.diff(points, axis=0)
    distances = np.r_[0, np.cumsum(np.linalg.norm(segments, axis=1))]
    # Estimate the direction over a physical neighborhood to smooth guide corners.
    normals = np.column_stack([
        np.interp(distances + 3, distances, axis)
        - np.interp(distances - 3, distances, axis)
        for axis in points.T
    ])
    stalled = np.flatnonzero(np.linalg.norm(normals, axis=1) < 1e-6)
    normals[stalled] = segments[np.minimum(stalled, len(segments) - 1)]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Keep the same reference axis throughout the course to avoid sudden roll.
    reference = np.eye(3)[np.argmin(np.abs(normals).max(axis=0))]
    horizontal = np.cross(reference, normals)
    horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
    vertical = np.cross(normals, horizontal)
    frames = np.repeat(np.eye(4)[None], len(points), axis=0)
    frames[:, :3, 0] = horizontal
    frames[:, :3, 1] = vertical
    frames[:, :3, 2] = normals
    frames[:, :3, 3] = points
    if previous is not None:
        # Keep the planes already visited and carry their orientation onto the chosen branch.
        frames[:len(previous)] = previous
        for index in range(len(previous), len(frames)):
            normal = normals[index]
            across = np.cross(frames[index - 1, :3, 1], normal)
            if np.linalg.norm(across) < 1e-6:
                across = frames[index - 1, :3, 0]
                across = across - normal * (across @ normal)
            across /= np.linalg.norm(across)
            frames[index, :3, 0] = across
            frames[index, :3, 1] = np.cross(normal, across)
    return frames


def closest_on_path(path, ras):
    if len(path) == 1:
        return path[0]
    segments = np.diff(path, axis=0)
    fraction = np.clip(
        np.sum((ras - path[:-1]) * segments, axis=1)
        / np.maximum(np.sum(segments * segments, axis=1), 1e-9),
        0,
        1,
    )
    candidates = path[:-1] + fraction[:, None] * segments
    return candidates[np.argmin(np.linalg.norm(candidates - ras, axis=1))]


class ArteryMap:
    def __init__(self, case, meta, dimensions):
        self.case = Path(case)
        self.affine = np.asarray(meta["nativeIJKToRAS"], dtype=float)
        self.dimensions = list(dimensions)
        self.uid = meta["seriesUID"]
        self.preferred = None
        self.guides = []
        self.pins = []
        self.reload()

    def geometry(self):
        return dict(
            version=1,
            seriesUID=self.uid,
            dimensions=self.dimensions,
            nativeIJKToRAS=self.affine.tolist(),
        )

    def read(self, filename):
        file = self.case / filename
        if not file.exists():
            return {}
        data = json.loads(file.read_text())
        if (
            data["version"] != 1
            or data["seriesUID"] != self.uid
            or data["dimensions"] != self.dimensions
            or not np.allclose(data["nativeIJKToRAS"], self.affine, rtol=0, atol=0.001)
        ):
            raise ValueError(f"{filename} belongs to a different source grid")
        for rec in data["landmarks"]:
            pts = np.asarray([rec["focusIJK"], *rec["pathIJK"]], float)
            if (
                pts.ndim != 2
                or pts.shape[1] != 3
                or not np.isfinite(pts).all()
                or np.any(pts < 0)
                or np.any(pts > np.asarray(self.dimensions) - 1)
                or not rec["pathIJK"]
                or not rec["id"]
                or not rec["name"].strip()
                or not 0 < rec["radiusMm"] <= 20
            ):
                raise ValueError(f"Invalid landmark in {filename}")
            if "downstream" in rec:
                children = rec["downstream"]
                if (
                    not isinstance(children, list)
                    or len(children) > 2
                    or any(not isinstance(key, str) or not key or key == rec["id"] for key in children)
                    or len(set(children)) != len(children)
                ):
                    raise ValueError(f"Invalid downstream branches in {filename}")
        return data

    def reload(self):
        source = self.read("artery-landmarks.json")
        self.preferred = source.get("preferred")
        self.guides = source.get("landmarks", [])
        self.pins = self.read("anatomy-pins.json").get("landmarks", [])
        self.records = self.guides + self.pins
        self.paths = {r["id"]: self.world(r["pathIJK"]) for r in self.records}
        self.subsegmental_guides = [
            r for r in self.guides if r.get("level") == "subsegmental" and len(r["pathIJK"]) > 1
        ]

    def world(self, points):
        return np.asarray(points, float) @ self.affine[:3, :3].T + self.affine[:3, 3]

    def course_steps(self, record):
        points = np.asarray(record["pathIJK"], float)
        steps = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            delta = end - start
            length = np.linalg.norm(self.affine[:3, :3] @ delta)
            # Visit each crossed axial slice and keep moving along in-plane sections.
            count = int(np.ceil(max(abs(delta[2]), length / 3)))
            steps.extend(start + delta * (i / count) for i in range(1, count + 1))
        return self.world(steps)

    def nearby(self, ras, vessel=None):
        scored = []
        for rec in self.records:
            if (
                vessel
                and vessel != rec.get("vessel", "artery")
                and rec not in self.pins
            ):
                continue
            path = self.paths[rec["id"]]
            distance = np.linalg.norm(closest_on_path(path, ras) - ras)
            if distance <= rec["radiusMm"]:
                # A junction, or a user's pin, explicitly marks a small neighborhood.
                if len(path) == 1:
                    return rec
                scored.append((distance, rec))
        scored.sort(key=lambda item: item[0])
        if not scored or (len(scored) > 1 and scored[1][0] - scored[0][0] < 0.65):
            return None
        return scored[0][1]

    def subsegmental_pick(self, ras, margin=0, supports=None):
        """Resolve a click to a single mapped daughter and its centerline position."""
        scored = []
        for record in self.subsegmental_guides:
            center = closest_on_path(self.paths[record["id"]], ras)
            distance = np.linalg.norm(center - ras)
            if distance <= record["radiusMm"] + margin and (
                supports is None or supports(center)
            ):
                scored.append((distance, record, center))
        scored.sort(key=lambda hit: hit[0])
        # The shared lumen at a fork cannot identify either daughter on its own.
        if not scored or (len(scored) > 1 and scored[1][0] - scored[0][0] < 0.35):
            return None
        return scored[0][1:]

    def edit_pin(self, name=None, ijk=None, remove_id=None):
        key = remove_id or "pin-" + uuid4().hex
        with (self.case / ".anatomy-pins.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read("anatomy-pins.json") or dict(self.geometry(), landmarks=[])
            pins = data["landmarks"]
            if remove_id:
                pins[:] = [p for p in pins if p["id"] != remove_id]
            else:
                pt = np.asarray(ijk, float)
                if (
                    pt.shape != (3,)
                    or not np.all(np.isfinite(pt))
                    or np.any(pt < 0)
                    or np.any(pt > np.asarray(self.dimensions) - 1)
                ):
                    raise ValueError("Select a location within the source CT")
                if not name.strip():
                    raise ValueError("Enter a name for the pin")
                pins.append(
                    dict(
                        id=key,
                        name=name.strip()[:72],
                        radiusMm=2.0,
                        focusIJK=pt.tolist(),
                        pathIJK=[pt.tolist()],
                        description="User pin",
                    )
                )
            with NamedTemporaryFile(mode="w", dir=self.case, delete=False) as temp:
                try:
                    json.dump(data, temp)
                    temp.flush()
                    os.fsync(temp.fileno())
                    os.replace(temp.name, self.case / "anatomy-pins.json")
                finally:
                    Path(temp.name).unlink(missing_ok=True)
        self.reload()
        return key


class VesselLabels:
    """Native-grid artery/vein estimates packed to one bit per voxel per class."""

    def __init__(self, case, affine, dimensions):
        from grid import load_grid

        self.shape = tuple(reversed(dimensions))
        self.masks = {}
        for kind, file in (
            ("artery", "lung_arteries.nii.gz"),
            ("vein", "lung_veins.nii.gz"),
        ):
            path = Path(case) / "vessels" / file
            if path.is_file():
                try:
                    mask, geometry, _ = load_grid(path)
                except (OSError, RuntimeError):
                    continue
                if mask.shape != self.shape or not np.allclose(
                    geometry, affine, atol=0.02, rtol=0
                ):
                    continue
                self.masks[kind] = np.packbits(mask != 0, axis=2, bitorder="little")

    def at(self, ijk):
        x, y, z = np.rint(ijk).astype(int)
        if not (
            0 <= x < self.shape[2] and 0 <= y < self.shape[1] and 0 <= z < self.shape[0]
        ):
            return None
        hits = [
            kind
            for kind, bits in self.masks.items()
            if bits[z, y, x // 8] & (1 << (x % 8))
        ]
        return hits[0] if len(hits) == 1 else None

    def arterial_profile(self, ijk):
        """Return the connected axial artery mask at the guide, and its I/J origin."""
        i, j, k = np.rint(ijk).astype(int)
        plane = np.unpackbits(
            self.masks["artery"][k], axis=1, count=self.shape[2], bitorder="little"
        )
        return self.connected_profile(plane, i, j)

    def section_profile(self, grid_to_ijk, radius):
        """Sample the same artery mask on a plane centered at the guide."""
        size = 2 * radius + 1
        j, i = np.ogrid[:size, :size]
        x, y, z = (
            np.rint(origin + i * across + j * down).astype(int)
            for across, down, origin in grid_to_ijk[:3, (0, 1, 3)]
        )
        depth, height, width = self.shape
        inside = (0 <= x) & (x < width) & (0 <= y) & (y < height) & (0 <= z) & (z < depth)
        x, y, z = x[inside], y[inside], z[inside]
        plane = np.zeros((size, size), dtype=bool)
        plane[inside] = self.masks["artery"][z, y, x // 8] & (1 << (x % 8))
        return self.connected_profile(plane, radius, radius)

    @staticmethod
    def connected_profile(plane, i, j):
        from scipy import ndimage

        if not plane[j, i]:
            return None
        # Background outside the occupied rows and columns cannot join the component.
        occupied_rows = np.flatnonzero(plane.any(axis=1))
        occupied_cols = np.flatnonzero(plane.any(axis=0))
        row_start, col_start = occupied_rows[0], occupied_cols[0]
        cropped = plane[row_start : occupied_rows[-1] + 1, col_start : occupied_cols[-1] + 1]
        labels, _ = ndimage.label(cropped, structure=np.ones((3, 3)))
        height, width = plane.shape
        component = labels[j % height - row_start, i % width - col_start]
        rows, cols = ndimage.find_objects(labels)[component - 1]
        return labels[rows, cols] == component, col_start + cols.start, row_start + rows.start
