"""
Варп BOLD Z-карты (T2) -> MNI для каждого субъекта.
Нужно для BOLD-gate в reliability-анализе.
"""
import os, glob
import numpy as np, nibabel as nib
from nilearn.image import resample_to_img
import ants

BASE = "ds004873/derivatives"
SUBJECTS = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055"]

def bold_t2(s):
    p = f"{BASE}/{s}/func/{s}_1stlevel_calccontrol_space-T2.nii.gz"
    return p if os.path.exists(p) else None
def t1w_ref(s):
    p = f"{BASE}/{s}/anat/{s}_desc-fmriprep_T1w.nii.gz"
    return p if os.path.exists(p) else None
def orig_mni_ref(s):  # эталон MNI-сетки (совпадает с CMRO2 MNI)
    p = f"{BASE}/{s}/qmri/{s}_task-calc_space-MNI152_desc-orig_cmro2.nii.gz"
    return p if os.path.exists(p) else None
def find_h5(s):
    c = glob.glob(f"{BASE}/{s}/anat/*from-T1w*to-MNI*xfm.h5")
    if not c: c = glob.glob(f"{BASE}/{s}/**/*from-T1w*to-MNI*.h5", recursive=True)
    return c[0] if c else None

done, skipped = [], []
for s in SUBJECTS:
    bold, t1, ref, h5 = bold_t2(s), t1w_ref(s), orig_mni_ref(s), find_h5(s)
    miss = [n for n,p in [("bold",bold),("t1w",t1),("orig_mni",ref),("h5",h5)] if not p]
    if miss:
        print(f"[SKIP] {s}: нет {miss}"); skipped.append(s); continue
    try:
        # 1) T2 -> T1w: NaN->0 до ресемпла
        im = nib.load(bold); d = im.get_fdata(dtype=np.float32)
        d[~np.isfinite(d)] = 0.0
        vmask_src = (np.isfinite(im.get_fdata(dtype=np.float32))).astype(np.float32)

        t1_nib = nib.load(t1)
        bold_t1 = resample_to_img(
            nib.Nifti1Image(d, im.affine, im.header), t1_nib, interpolation="continuous")
        vmask_t1 = resample_to_img(
            nib.Nifti1Image(vmask_src, im.affine, im.header), t1_nib, interpolation="nearest")

        a = bold_t1.get_fdata().astype(np.float32)
        vm = vmask_t1.get_fdata().astype(np.float32) > 0.5
        a[~vm] = 0.0

        nib.save(nib.Nifti1Image(a, bold_t1.affine, bold_t1.header), f"/tmp/{s}_bold_d.nii.gz")

        # 2) T1w -> MNI
        fixed = ants.image_read(ref)
        wd = ants.apply_transforms(fixed, ants.image_read(f"/tmp/{s}_bold_d.nii.gz"), [h5], "linear")
        out = f"{BASE}/{s}/func/{s}_task-calc_space-MNI152_desc-Zstat_bold.nii.gz"
        ants.image_write(wd, out)
        arr = wd.numpy(); valid = arr[np.isfinite(arr) & (arr > 0)]
        print(f"[OK] {s}  p50={np.percentile(valid,50):.2f}  max={valid.max():.2f}")
        done.append(s)
    except Exception as e:
        print(f"[ERR] {s}: {e}"); skipped.append(s)

print(f"\nГотово: {len(done)}  пропущено: {len(skipped)}")