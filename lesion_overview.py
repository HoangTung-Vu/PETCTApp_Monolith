from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

DATA_DIR = Path(
    "/home/hoangtungvm/UET/RESEARCH/MAE_AD_PETCT/data/autopet_samples/"
    "PETCT_07574bfa00/"
    "04-20-2003-NA-PET-CT Ganzkoerper  primaer mit KM-91235"
)
CT_PATH = DATA_DIR / "CTres.nii.gz"
SUV_PATH = DATA_DIR / "SUV.nii.gz"
SEG_PATH = DATA_DIR / "SEG.nii.gz"
OUT_PATH = Path(__file__).parent / "lesion_overview.png"

SUV_VMAX = 10.0
CT_WL, CT_WW = 40, 400
THIN_MIP_HALF = 0


def display(slice2d):
    return slice2d.T


def voxel_spacing(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def largest_lesion_centroid(seg):
    labeled, n = ndimage.label(seg > 0)
    if n == 0:
        raise RuntimeError("No lesion in SEG.")
    sizes = ndimage.sum(seg > 0, labeled, index=np.arange(1, n + 1))
    largest_id = int(np.argmax(sizes)) + 1
    mask = labeled == largest_id
    cy = int(round(np.argwhere(mask).mean(axis=0)[1]))
    return mask, cy


def main():
    ct_img = nib.load(str(CT_PATH))
    suv_img = nib.load(str(SUV_PATH))
    seg_img = nib.load(str(SEG_PATH))
    ct = ct_img.get_fdata()
    suv = suv_img.get_fdata()
    seg = seg_img.get_fdata()
    assert ct.shape == suv.shape == seg.shape

    _, cy = largest_lesion_centroid(seg)
    all_lesions = seg > 0

    suv_clipped = np.clip(suv, 0, SUV_VMAX)
    suv_mip = display(suv_clipped.max(axis=1))
    mask_mip = display(all_lesions.max(axis=1))

    y0 = max(cy - THIN_MIP_HALF, 0)
    y1 = min(cy + THIN_MIP_HALF + 1, ct.shape[1])
    ct_lo, ct_hi = CT_WL - CT_WW / 2, CT_WL + CT_WW / 2
    ct_thin_mip = display(np.clip(ct[:, y0:y1, :], ct_lo, ct_hi).max(axis=1))

    def crop_sides(img, frac=1 / 7):
        w = img.shape[1]
        cut = int(round(w * frac))
        return img[:, cut:w - cut]

    suv_mip = crop_sides(suv_mip)
    mask_mip = crop_sides(mask_mip)
    ct_thin_mip = crop_sides(ct_thin_mip)

    dx_suv, _, dz_suv = voxel_spacing(suv_img.affine)
    dx_ct, _, dz_ct = voxel_spacing(ct_img.affine)
    extent_suv = (0, suv_mip.shape[1] * dx_suv, 0, suv_mip.shape[0] * dz_suv)
    extent_ct = (0, ct_thin_mip.shape[1] * dx_ct, 0, ct_thin_mip.shape[0] * dz_ct)

    overlay_cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.6)])

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    axes[0].imshow(suv_mip, cmap="jet", vmin=0, vmax=SUV_VMAX,
                   extent=extent_suv, aspect="equal", origin="lower")
    axes[0].imshow(mask_mip, cmap=overlay_cmap,
                   extent=extent_suv, aspect="equal", origin="lower")
    axes[0].set_axis_off()

    axes[1].imshow(ct_thin_mip, cmap="gray", vmin=ct_lo, vmax=ct_hi,
                   extent=extent_ct, aspect="equal", origin="lower")
    axes[1].set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
