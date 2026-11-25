import numpy as np
import SimpleITK as sitk


def load_grid(file):
    image = sitk.ReadImage(str(file))
    axes = np.array(image.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = axes @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine[:2] *= -1
    return sitk.GetArrayFromImage(image), affine, image


def save_mask(mask, reference, file):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(file), useCompression=True)
