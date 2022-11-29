from options import OptionsDcm2Nii
import os
import numpy as np
import nibabel as nib
import pydicom


def mosaic_dcm_to_nii(in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i, dcm_file in enumerate(
        sorted([f for f in os.listdir(in_folder) if f.endswith(".dcm")])
    ):
        dcm = pydicom.dcmread(os.path.join(in_folder, dcm_file))
        nii = _mosaic_dcm_to_nii(dcm)
        nib.save(nii, os.path.join(out_folder, "%04d.nii.gz" % i))


def _mosaic_dcm_to_nii(dcm):
    img = dcm.pixel_array.astype(np.float32)
    acqmatsize = (
        dcm.AcquisitionMatrix
    )  # AcquisitionMatrix: [frequency rows, frequency columns, phase rows, phase columns]
    img = _demosaic(
        img,
        acqmatsize[0],
        acqmatsize[0],
        (dcm.Rows * dcm.Columns) // (acqmatsize[0] * acqmatsize[0]),
    )
    resolution = [
        float(dcm.PixelSpacing[0]),
        float(dcm.PixelSpacing[1]),
        float(dcm.SpacingBetweenSlices),
    ]
    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = resolution[i]
    return nib.Nifti1Image(img, affine)


def _demosaic(mosaic, x, y, z):
    data = np.zeros((x, y, z), dtype=mosaic.dtype)
    x, y, z = data.shape
    n = np.ceil(np.sqrt(z))
    dim = int(np.sqrt(np.prod(mosaic.shape)))
    mosaic = mosaic.reshape(dim, dim)
    for idx in range(z):
        x_idx = int(np.floor(idx / n)) * x
        y_idx = int(idx % n) * y
        data[..., idx] = mosaic[x_idx : x_idx + x, y_idx : y_idx + y]
    return data


if __name__ == "__main__":
    opts = OptionsDcm2Nii().parse()
    mosaic_dcm_to_nii(opts.rawdata_path, opts.output_path)
