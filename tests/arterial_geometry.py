"""Regression for flagging external tissue that touches an arterial mask."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from arterial_evidence import grade


def check():
    spacing = np.array([1.3, 0.7, 0.7])
    z, y, x = np.indices((30, 64, 64)) * spacing[:, None, None, None]
    vessel = (x - 21) ** 2 + (y - 21) ** 2 <= 6**2
    inner = (x - 21) ** 2 + (y - 21) ** 2 + (z - 13) ** 2 <= 2.5**2
    outer = (x - 29) ** 2 + (y - 21) ** 2 + (z - 26) ** 2 <= 2.5**2
    assert (outer & vessel).any()
    ct = np.full(vessel.shape, -750, dtype=np.int16)
    ct[vessel] = 270
    ct[inner | outer] = 55
    proposals = [{"centerIJK": [30, 30, 10]}, {"centerIJK": [41, 30, 20]}]
    tiers, regions = grade(ct, vessel, inner | outer, proposals, spacing)
    assert regions[0]["supportedVoxels"] > 0
    assert tiers[tuple(regions[0]["evidenceIJK"][::-1])] == 1
    assert np.all(tiers[outer] == 2)
    assert regions[1]["supportedVoxels"] == 0
    assert np.array_equal(tiers > 0, inner | outer)
    assert not ((tiers == 1) & ~vessel).any()
    empty, records = grade(ct, vessel, np.zeros_like(vessel), [], spacing)
    assert not empty.any() and records == []


if __name__ == "__main__":
    check()
    print("Intraluminal, wall-contact and empty proposals passed on anisotropic CT.")
