from options import OptionsSplitNii
import os
import numpy as np
import nibabel as nib


def split_nii(in_folder, out_folder, starts_with_even):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i, nii_file in enumerate(
        sorted(
            [
                f
                for f in os.listdir(in_folder)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
        )
    ):
        nii = nib.load(os.path.join(in_folder, nii_file))
        affine = nii.affine

        def split(img, even):
            img_s = np.copy(img)
            for j in range(1 if even else 0, img.shape[-1], 2):
                if j > 0 and j < img.shape[-1] - 1:
                    img_s[:, :, j] = (img[:, :, j - 1] + img[:, :, j + 1]) / 2
            return img_s

        img = nii.get_fdata()
        img_even = split(img, even=True)
        img_odd = split(img, even=False)
        if starts_with_even:
            img0, img1 = img_even, img_odd
        else:
            img0, img1 = img_odd, img_even
        nii0 = nib.Nifti1Image(img0, affine)
        nib.save(nii0, os.path.join(out_folder, "%04d.nii.gz" % (2 * i)))
        nii1 = nib.Nifti1Image(img1, affine)
        nib.save(nii1, os.path.join(out_folder, "%04d.nii.gz" % (2 * i + 1)))


if __name__ == "__main__":
    opts = OptionsSplitNii().parse()
    split_nii(opts.rawdata_path, opts.output_path, not opts.starts_with_odd)
