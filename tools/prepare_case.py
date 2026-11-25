"""Propose intraluminal low-attenuation candidates using one CT and its arterial mask."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arterial_evidence import grade
from forest import Forest
from grid import load_grid, save_mask
from scipy import ndimage as nd


def flag(ct, artery, spacing):
    from appearance import predict

    scores = predict(ct, artery, spacing, Forest())
    # Grow a connected lower-threshold outline only from strongly scored voxels.
    labels, _ = nd.label(scores >= 150, np.ones((3, 3, 3), bool))
    counts = np.bincount(labels.ravel())
    entries = []
    voxel_ml = float(np.prod(spacing)) * 0.001
    for i, sl in enumerate(nd.find_objects(labels), 1):
        if counts[i] * voxel_ml < 0.012 or sl[0].stop - sl[0].start < 2:
            continue
        region = labels[sl] == i
        values = scores[sl][region]
        if np.sum(values >= 210) < max(3, round(0.002 / voxel_ml)):
            continue
        q = np.argwhere(region) + np.array([s.start for s in sl])
        if not np.any(artery[tuple(q.T)]):
            continue
        center = q[
            np.argmin(np.linalg.norm((q - np.median(q, axis=0)) * spacing, axis=1))
        ]
        entries.append(
            (
                i,
                {
                    "centerIJK": center[::-1].tolist(),
                    "voxels": int(counts[i]),
                    "volumeMl": float(counts[i] * voxel_ml),
                },
            )
        )
    entries.sort(key=lambda pair: -pair[1]["voxels"])
    return np.isin(labels, [i for i, _ in entries]), [e for _, e in entries]


def surface(mask, affine, out):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    img = vtk.vtkImageData()
    img.SetDimensions(*mask.shape[::-1])
    scalars = numpy_to_vtk(
        np.ascontiguousarray(mask, dtype=np.uint8).ravel(), deep=True
    )
    img.GetPointData().SetScalars(scalars)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(img)
    contour.SetValue(0, 0.5)
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(10)
    smooth.NormalizeCoordinatesOn()
    smooth.SetPassBand(0.12)
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputConnection(smooth.GetOutputPort())
    dec.SetTargetReduction(0.52)
    trans = vtk.vtkTransform()
    trans.SetMatrix(affine.ravel())
    tr = vtk.vtkTransformPolyDataFilter()
    tr.SetInputConnection(dec.GetOutputPort())
    tr.SetTransform(trans)
    write = vtk.vtkXMLPolyDataWriter()
    write.SetInputConnection(tr.GetOutputPort())
    write.SetFileName(str(out))
    if not write.Write():
        raise IOError(out)


def prepare(case, regrade=False):
    ct, affine, reference = load_grid(case / "ct.nii.gz")
    mask, geometry, _ = load_grid(case / "vessels/lung_arteries.nii.gz")
    if mask.shape != ct.shape or not np.allclose(affine, geometry, atol=0.02):
        raise ValueError("Artery geometry mismatch")
    artery = mask > 0
    spacing = np.array(reference.GetSpacing()[::-1])
    if regrade:
        mask, old_geometry, _ = load_grid(case / "automatic.nii.gz")
        if mask.shape != ct.shape or not np.allclose(old_geometry, affine, atol=0.02):
            raise ValueError("Proposal geometry mismatch")
        mask = mask > 0
        regions = json.loads((case / "candidates.json").read_text())["regions"]
    else:
        mask, regions = flag(ct, artery, spacing)
    tiers, regions = grade(ct, artery, mask, regions, spacing)
    with TemporaryDirectory(dir=case) as temp:
        work = Path(temp)
        if not regrade:
            save_mask(mask, reference, work / "automatic.nii.gz")
            os.replace(work / "automatic.nii.gz", case / "automatic.nii.gz")
        save_mask(tiers, reference, work / "evidence.nii.gz")
        digest = hashlib.sha256((work / "evidence.nii.gz").read_bytes()).hexdigest()[
            :20
        ]
        filename = f"evidence-{digest}.nii.gz"
        os.replace(work / "evidence.nii.gz", case / filename)
        if not (case / "artery-ras.vtp").exists():
            surface(artery, affine, work / "artery.vtp")
            os.replace(work / "artery.vtp", case / "artery-ras.vtp")
        (work / "candidates.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "method": "fumpe-appearance-arterial-evidence-v2",
                    "evidenceFile": filename,
                    "regions": regions,
                }
            )
        )
        os.replace(work / "candidates.json", case / "candidates.json")
    print(case.name, len(regions), "candidate regions", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument(
        "--regrade", action="store_true", help="Use existing appearance proposals"
    )
    args = p.parse_args()
    prepare(args.case, args.regrade)
