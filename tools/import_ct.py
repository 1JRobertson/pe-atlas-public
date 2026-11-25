"""Import exactly one conventional axial CTPA series, preserving source-matrix order."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def import_series(folder, output, title, series_number=None):
    tags = [
        "SeriesInstanceUID",
        "SeriesNumber",
        "InstanceNumber",
        "SOPInstanceUID",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "Rows",
        "Columns",
    ]
    series = {}
    for file in folder.rglob("*"):
        if not file.is_file() or file.name.startswith("."):
            continue
        try:
            d = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=tags)
        except pydicom.errors.InvalidDicomError:
            continue
        if not all(hasattr(d, k) for k in tags):
            continue
        if series_number is None or int(d.SeriesNumber) == series_number:
            series.setdefault(d.SeriesInstanceUID, []).append((d, file))
    if len(series) != 1:
        raise ValueError(
            f"Expected one series, found {len(series)}. Select a SeriesNumber."
        )
    slices = next(iter(series.values()))
    first = slices[0][0]
    row_dir, col_dir = np.array(first.ImageOrientationPatient).reshape(2, 3)
    normal = np.cross(row_dir, col_dir)
    if abs(normal[2]) < 0.8:
        raise ValueError("An axial acquisition is required.")
    # Preserve the source in superior-to-inferior order, regardless of file naming.
    slices.sort(key=lambda s: -float(s[0].ImagePositionPatient[2]))
    pos = np.array([d.ImagePositionPatient for d, _ in slices], dtype=float)
    step = np.median(np.diff(pos, axis=0), axis=0)
    if (
        np.linalg.norm(step) < 0.1
        or np.max(np.abs(pos - (pos[0] + np.arange(len(pos))[:, None] * step))) > 0.12
    ):
        raise ValueError("Missing, repeated, or irregularly spaced slices.")
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack(
        (row_dir * first.PixelSpacing[1], col_dir * first.PixelSpacing[0], step)
    )
    affine[:3, 3] = pos[0]
    affine[:2] *= -1  # DICOM LPS to Slicer RAS
    volume = np.empty((len(slices), first.Rows, first.Columns), dtype=np.int16)
    refs = []
    for index, (d, file) in enumerate(slices):
        if (
            (d.Rows, d.Columns) != volume.shape[1:]
            or not np.allclose(d.ImageOrientationPatient, first.ImageOrientationPatient)
            or not np.allclose(d.PixelSpacing, first.PixelSpacing)
        ):
            raise ValueError("Inconsistent in-plane geometry.")
        original = pydicom.dcmread(file)
        hu = original.pixel_array.astype(np.float32) * float(
            original.get("RescaleSlope", 1)
        ) + float(original.get("RescaleIntercept", 0))
        volume[index] = np.rint(hu).astype(np.int16)
        refs.append(
            {
                "instance": int(d.InstanceNumber),
                "uid": str(d.SOPInstanceUID),
                "image": file.name,
                "positionLPS": list(map(float, d.ImagePositionPatient)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.T, affine), output / "ct.nii.gz")
    (output / "source.json").write_text(
        json.dumps(
            {
                "title": title,
                "series": int(first.SeriesNumber),
                "seriesUID": str(first.SeriesInstanceUID),
                "nativeIJKToRAS": affine.tolist(),
                "slices": refs,
            },
            indent=1,
        )
    )
    print(f"{title}: {len(refs)} original slices", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--series", type=int)
    a = p.parse_args()
    import_series(a.folder, a.output, a.title, a.series)
