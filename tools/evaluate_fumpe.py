"""Compare saved automatic predictions to expert FUMPE masks, without modifying cases."""

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as nd
from scipy.io import loadmat


def reference_mask(case, reference):
    meta = json.loads((case / "source.json").read_text())
    gt = loadmat(reference)["Mask"]
    # MATLAB rows/columns and original DICOM filename index, per the FUMPE release.
    order = [
        int(re.fullmatch(r"D(\d+)\.dcm", s["image"], re.I).group(1)) - 1
        for s in meta["slices"]
    ]
    if sorted(order) != list(range(gt.shape[2])):
        raise ValueError("Reference slice identity mismatch")
    truth = gt.transpose(2, 0, 1)[order].astype(bool)
    return truth


def evaluate(case, reference):
    truth = reference_mask(case, reference)
    pred = np.asarray(nib.load(case / "automatic.nii.gz").dataobj).T.astype(bool)
    if truth.shape != pred.shape:
        raise ValueError("Reference grid mismatch")
    structure = np.ones((3, 3, 3), bool)
    blobs, num = nd.label(pred, structure)
    expert, nref = nd.label(truth, structure)
    overlap = int((pred & truth).sum())
    npred = int(pred.sum())
    ntruth = int(truth.sum())
    matched = len(np.unique(blobs[truth & pred]))
    found = len(np.unique(expert[truth & pred]))
    return {
        "case": case.name,
        "source_slices": len(truth),
        "expert_regions": nref,
        "candidate_regions": num,
        "expert_regions_with_any_overlap": found,
        "candidate_regions_without_overlap": num - matched,
        "voxel_dice": round(2 * overlap / (npred + ntruth), 4)
        if npred + ntruth
        else None,
        "voxel_precision": round(overlap / npred, 4) if npred else None,
        "voxel_recall": round(overlap / ntruth, 4) if ntruth else None,
    }


def compare_tiers(case, reference):
    """Count the same proposal IDs before/after and measure actual voxel overlap."""
    truth = reference_mask(case, reference)
    proposals = nib.load(case / "automatic.nii.gz")
    manifest = json.loads((case / "candidates.json").read_text())
    evidence = nib.load(case / manifest["evidenceFile"])
    raw = np.asarray(proposals.dataobj).T.astype(bool)
    tiers = np.asarray(evidence.dataobj).T
    if (
        not np.allclose(evidence.affine, proposals.affine)
        or truth.shape != raw.shape
        or not np.array_equal(raw, tiers > 0)
    ):
        raise ValueError("Evidence and proposal geometry/coverage mismatch")
    structure = np.ones((3, 3, 3), bool)
    proposals, count = nd.label(raw, structure)
    expert, expert_count = nd.label(truth, structure)
    if count != len(manifest["regions"]):
        raise ValueError("Mismatch between proposal IDs and source components")

    def metrics(mask):
        ids = np.unique(proposals[mask])
        overlap = mask & truth
        matched = np.unique(proposals[overlap])
        tp = int(overlap.sum())
        p = int(mask.sum())
        t = int(truth.sum())
        return {
            "flagged": len(ids),
            "without_expert_overlap": len(ids) - len(matched),
            "expert_regions_touched": len(np.unique(expert[overlap])),
            "overlap_voxels": tp,
            "predicted_voxels": p,
            "expert_voxels": t,
            "dice": round(2 * tp / (p + t), 4) if p + t else None,
            "precision": round(tp / p, 4) if p else None,
            "recall": round(tp / t, 4) if t else None,
        }

    return {
        "case": case.name,
        "expert_regions": expert_count,
        "voxel_mm3": float(abs(np.linalg.det(evidence.affine[:3, :3]))),
        "original": metrics(raw),
        "red": metrics(tiers == 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--compare-tiers", action="store_true")
    a = ap.parse_args()
    method = compare_tiers if a.compare_tiers else evaluate
    print(json.dumps(method(a.case, a.reference), indent=2))
