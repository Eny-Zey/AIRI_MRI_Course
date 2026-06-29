"""
PSC из 4D BOLD (T2) -> варп в MNI.
Поддерживает два протокола: ID<56 (400 томов) и ID>55 (200 томов).
"""
import os, glob, re
import numpy as np, nibabel as nib
from nilearn.image import resample_to_img
import ants

BASE = "ds004873/derivatives"

# Все субъекты включая новых
SUBJECTS_LT56 = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055"]

SUBJECTS_GT55 = [
    "sub-p058","sub-p059","sub-p060","sub-p061","sub-p063",
    "sub-p064","sub-p065","sub-p066","sub-p067","sub-p068"]

def idx(ranges):
    out = []
    for a,b in ranges: out.extend(range(a,b))
    return out

# Временные окна из ноутбука критиков
CONTROL_BASE_LT56 = idx([(80,100),(205,225),(280,300),(355,375)])
CALC_WIN_LT56     = idx([(55,75),(155,175),(255,275),(305,325)])
CONTROL_BASE_GT55 = idx([(5,25),(55,75),(105,125),(155,175)])
CALC_WIN_GT55     = idx([(30,50),(80,100),(130,150),(180,200)])

def perc_change(func_arr, baseline_3d, eps=1e-6):
    b = baseline_3d[..., None].astype(np.float32)
    pc = 100.0 * (func_arr.astype(np.float32) - b) / (np.abs(b) + eps)
    invalid = (~np.isfinite(baseline_3d)) | (baseline_3d <= eps)
    pc[invalid, :] = np.nan
    return pc

def find_h5(s):
    c = glob.glob(f"{BASE}/{s}/anat/*from-T1w*to-MNI*xfm.h5")
    if not c: c = glob.glob(f"{BASE}/{s}/**/*from-T1w*to-MNI*.h5", recursive=True)
    return c[0] if c else None

def orig_mni_ref(s):
    # для ID>55 calc orig MNI нет — берём control
    for cond in ["calc","control"]:
        p = f"{BASE}/{s}/qmri/{s}_task-{cond}_space-MNI152_desc-orig_cmro2.nii.gz"
        if os.path.exists(p): return p
    return None

def process(s, ctrl_base_idx, calc_idx):
    bold_t2 = f"{BASE}/{s}/func/{s}_task-all_space-T2_desc-preproc_bold.nii.gz"
    out_mni  = f"{BASE}/{s}/func/{s}_task-calccontrol_space-MNI152_res-2_desc-percchange_bold.nii.gz"

    if os.path.exists(out_mni) and os.path.getsize(out_mni) > 10000:
        print(f"[SKIP] {s}: уже есть"); return True

    if not os.path.exists(bold_t2) or os.path.getsize(bold_t2) < 1e6:
        print(f"[WAIT] {s}: 4D не скачан"); return False

    h5  = find_h5(s)
    ref = orig_mni_ref(s)
    t1w = f"{BASE}/{s}/anat/{s}_desc-fmriprep_T1w.nii.gz"

    miss = [n for n,p in [("h5",h5),("mni_ref",ref),("t1w",t1w)] if not p]
    if miss: print(f"[SKIP] {s}: нет {miss}"); return False

    try:
        print(f"[PROC] {s}...")
        img  = nib.load(bold_t2)
        data = img.get_fdata(dtype=np.float32)
        T    = data.shape[3]
        print(f"  shape: {data.shape}")

        # проверяем что индексы не выходят за границу
        max_idx = max(max(ctrl_base_idx), max(calc_idx))
        if max_idx >= T:
            print(f"  [ERR] индекс {max_idx} >= T={T}, пропускаем"); return False

        baseline = np.nanmedian(data[..., ctrl_base_idx], axis=3)
        pc       = perc_change(data, baseline)
        psc      = np.nanmedian(pc[..., calc_idx], axis=3)
        del data, pc

        # NaN→0 для интерполяции, маска отдельно
        psc_filled = psc.copy(); psc_filled[~np.isfinite(psc_filled)] = 0.0
        vmask = np.isfinite(psc).astype(np.float32)

        t1_nib   = nib.load(t1w)
        psc_t1   = resample_to_img(
            nib.Nifti1Image(psc_filled, img.affine, img.header),
            t1_nib, interpolation="continuous")
        vmask_t1 = resample_to_img(
            nib.Nifti1Image(vmask, img.affine, img.header),
            t1_nib, interpolation="nearest")

        a = psc_t1.get_fdata().astype(np.float32)
        a[vmask_t1.get_fdata() < 0.5] = 0.0
        nib.save(nib.Nifti1Image(a, psc_t1.affine), f"/tmp/{s}_psc.nii.gz")

        fixed = ants.image_read(ref)
        wd = ants.apply_transforms(fixed,
             ants.image_read(f"/tmp/{s}_psc.nii.gz"), [h5], "linear")
        ants.image_write(wd, out_mni)

        arr = wd.numpy(); v = arr[np.isfinite(arr) & (arr!=0)]
        print(f"  [OK] p1={np.percentile(v,1):.2f}  p50={np.percentile(v,50):.2f}  p99={np.percentile(v,99):.2f}")
        return True

    except Exception as e:
        print(f"  [ERR] {s}: {e}"); return False

done, waiting = [], []
for s in SUBJECTS_LT56:
    r = process(s, CONTROL_BASE_LT56, CALC_WIN_LT56)
    (done if r else waiting).append(s)

for s in SUBJECTS_GT55:
    r = process(s, CONTROL_BASE_GT55, CALC_WIN_GT55)
    (done if r else waiting).append(s)

print(f"\nГотово: {len(done)}  ждут: {len(waiting)} {waiting}")