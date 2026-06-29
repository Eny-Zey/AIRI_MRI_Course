"""
semi-quant-corrected CMRO2 (CALC): T2 -> T1w (nilearn, как в label_all)
-> MNI152 (.h5, ANTs). Кладём на сетку orig-MNI карт, чтобы воксельно
совпадало с orig control для reliability. Невалидные воксели -> NaN
(через варп маски валидности, без 0-заливки абсолютной карты).
Запуск: /home/eny-zey/miniconda3/envs/MRI/bin/python warp_semiquant_to_mni.py
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
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055",
    # новые субъекты (ID>55, только calc+control)
    "sub-p058","sub-p059","sub-p060","sub-p061","sub-p063",
    "sub-p064","sub-p065","sub-p066","sub-p067","sub-p068"]

def semi_t2(s):
    gz=f"{BASE}/{s}/qmri/{s}_task-calc_base-control_space-T2_desc-semi-quant-corrected_cmro2.nii.gz"
    return gz if os.path.exists(gz) else (gz[:-3] if os.path.exists(gz[:-3]) else None)
def t1w_ref(s): 
    p=f"{BASE}/{s}/anat/{s}_desc-fmriprep_T1w.nii.gz"; return p if os.path.exists(p) else None
def orig_mni_ref(s):
    # сначала calc, потом control — сетка MNI одинаковая
    for cond in ["calc", "control"]:
        p = f"{BASE}/{s}/qmri/{s}_task-{cond}_space-MNI152_desc-orig_cmro2.nii.gz"
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p
    return None
def find_h5(s):
    c = glob.glob(f"{BASE}/{s}/anat/*from-T1w*to-MNI*xfm.h5")
    if not c: c = glob.glob(f"{BASE}/{s}/**/*from-T1w*to-MNI*.h5", recursive=True)
    return c[0] if c else None

done, skipped = [], []
for s in SUBJECTS:
    semi, t1, ref, h5 = semi_t2(s), t1w_ref(s), orig_mni_ref(s), find_h5(s)
    miss = [n for n,p in [("semi",semi),("t1w",t1),("orig_mni",ref),("h5",h5)] if not p]
    if miss:
        print(f"[SKIP] {s}: нет {miss}"); skipped.append(s); continue

    # 1) T2 -> T1w (по заголовкам, continuous), inf/nan санитизация
    im = nib.load(semi); d = im.get_fdata(dtype=np.float32); d[~np.isfinite(d)] = np.nan
    d_filled = d.copy()
    d_filled[~np.isfinite(d_filled)] = 0.0          # NaN/inf -> 0 для интерполяции
    vmask_src = (np.isfinite(d) & (d > 0)).astype(np.float32)  # 1 там, где данные реальные

    t1_ref = nib.load(t1)
    semi_t1_d = resample_to_img(
        nib.Nifti1Image(d_filled, im.affine, im.header), t1_ref, interpolation="continuous")
    semi_t1_m = resample_to_img(
        nib.Nifti1Image(vmask_src, im.affine, im.header), t1_ref, interpolation="continuous")

    a = semi_t1_d.get_fdata().astype(np.float32)
    vmask = semi_t1_m.get_fdata().astype(np.float32) > 0.5
    a[~vmask] = 0.0        # 0 (не NaN) — ANTs тоже не любит NaN при варпе

    nib.save(nib.Nifti1Image(a, semi_t1_d.affine, semi_t1_d.header),
             f"/tmp/{s}_d.nii.gz")
    nib.save(nib.Nifti1Image(vmask.astype(np.float32), semi_t1_d.affine, semi_t1_d.header),
             f"/tmp/{s}_m.nii.gz")
    # 2) T1w -> MNI (.h5) на сетку orig-MNI; маску валидности варпим отдельно
    fixed = ants.image_read(ref)
    wd = ants.apply_transforms(fixed, ants.image_read(f"/tmp/{s}_d.nii.gz"), [h5], "linear")
    wm = ants.apply_transforms(fixed, ants.image_read(f"/tmp/{s}_m.nii.gz"), [h5], "linear")
    out_arr = wd.numpy(); out_arr[wm.numpy() < 0.5] = np.nan   # не выдумываем там, где данных нет
    out = f"{BASE}/{s}/qmri/{s}_task-calc_space-MNI152_desc-semi-quant-corrected_cmro2.nii.gz"
    ants.image_write(wd.new_image_like(out_arr), out)
    print(f"[OK] {s}  med={np.nanmedian(out_arr[np.isfinite(out_arr)&(out_arr>0)]):.0f}")
    done.append(s)

print(f"\nГотово: {len(done)}  пропущено: {len(skipped)} {skipped}")