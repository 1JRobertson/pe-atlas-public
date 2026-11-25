"""Fit the bundled small appearance classifier from eight CC BY 4.0 FUMPE cases."""

import argparse
from pathlib import Path
from forest import export_model
import numpy as np
import pydicom
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from appearance import features

TRAIN = (5, 8, 11, 14, 17, 20, 23, 26)


def samples(root, num, rng):
    name = f"PAT{num:03}"
    files = sorted((root / name).rglob("D*.dcm"))
    dc = [pydicom.dcmread(f) for f in files]
    assert all(int(f.stem[1:]) == i + 1 for i, f in enumerate(files))
    ct = np.stack(
        [
            np.rint(
                x.pixel_array * float(x.get("RescaleSlope", 1))
                + float(x.get("RescaleIntercept", 0))
            ).astype(np.int16)
            for x in dc
        ]
    )
    pos = np.array([d.ImagePositionPatient for d in dc])
    spacing = np.r_[
        np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)), dc[0].PixelSpacing
    ]
    truth = (
        loadmat(root / "Reference" / "label-edited" / (name + ".mat"))["Mask"]
        .transpose(2, 0, 1)
        .astype(bool)
    )
    assert truth.shape == ct.shape
    allpos = np.argwhere(truth)
    if not len(allpos):
        return None
    pos = allpos[rng.choice(len(allpos), min(5500, len(allpos)), replace=False)]
    sample_n = 6000
    near = allpos[rng.integers(len(allpos), size=14000)]
    near = (
        near + np.rint(rng.normal(0, 4.0, size=near.shape) / spacing).astype(int)
    ).clip(0, np.array(ct.shape) - 1)
    near = near[~truth[tuple(near.T)]]
    # Sample hard mimics near enhancing blood throughout the chest and near air interfaces.
    r = rng.integers(0, np.array(ct.shape), size=(260000, 3))
    val = ct[tuple(r.T)]
    bright = r[(val > 150) & (val < 570)]
    bloodnear = (
        bright + np.rint(rng.normal(0, 3.0, size=bright.shape) / spacing).astype(int)
    ).clip(0, np.array(ct.shape) - 1)
    bval = ct[tuple(bloodnear.T)]
    bloodnear = bloodnear[(bval > -40) & (bval < 210) & ~truth[tuple(bloodnear.T)]]
    rand = r[(val > -40) & (val < 210) & ~truth[tuple(r.T)]]
    sets = []
    for pts in [near, bloodnear, rand]:
        sets.append(pts[rng.choice(len(pts), min(sample_n, len(pts)), replace=False)])
    neg = np.concatenate(sets)
    data = np.concatenate([pos, neg])
    X = features(ct, data, spacing)
    y = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    print(name, len(pos), len(neg), flush=True)
    return X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    rng = np.random.default_rng(407)
    sets = [v for i in TRAIN if (v := samples(a.root, i, rng)) is not None]
    X = np.concatenate([v[0] for v in sets])
    y = np.concatenate([v[1] for v in sets])
    model = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=23,
        min_samples_leaf=65,
        l2_regularization=3,
        random_state=407,
    )
    with threadpool_limits(limits=4):
        model.fit(X, y)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    export_model(model, a.output)
