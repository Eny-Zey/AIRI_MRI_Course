"""
ДЕНЬ 1, шаг 1: маска статистически надёжных вокселей (метод критиков),
ОГРАНИЧЕНО маской мозга/серого вещества в MNI (group_mask).

Для каждого вокселя внутри маски собирает ΔCMRO2 по 29 субъектам,
t-тест против нуля, FDR-поправка. Надёжные = значимый эффект.

Запуск:
  /home/eny-zey/miniconda3/envs/MRI/bin/python compute_reliability_mask.py
"""
import os
import numpy as np
import nibabel as nib
from scipy import stats

BASE = "ds004873/derivatives"
OUT = "reliability"
GROUP_MASK = "ds004873/derivatives/masks/group_mask.nii.gz"
os.makedirs(OUT, exist_ok=True)

CMRO2_MIN_CTRL = 1e-3
FDR_Q = 0.05
MIN_SUBJ = 15   # минимум субъектов с данными в вокселе

SUBJECTS = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055",
]

def load(sub, cond):
    p = os.path.join(BASE, sub, "qmri",
        f"{sub}_task-{cond}_space-MNI152_desc-orig_cmro2.nii.gz")
    return nib.load(p)

# --- маска мозга/GM ---
gm = nib.load(GROUP_MASK).get_fdata() > 0
print(f"Маска мозга (group_mask): {int(gm.sum())} вокселей")

# --- собираем ΔCMRO2 ---
print("\nЗагрузка ΔCMRO2 (MNI) по 29 субъектам...")
ref = load(SUBJECTS[0], "calc")
shape = ref.shape; affine = ref.affine
stack = np.full((len(SUBJECTS),) + shape, np.nan, dtype=np.float32)

for i, sub in enumerate(SUBJECTS):
    calc = load(sub, "calc").get_fdata().astype(np.float32)
    ctrl = load(sub, "control").get_fdata().astype(np.float32)
    valid = (np.abs(ctrl) > CMRO2_MIN_CTRL) & gm
    d = np.full(shape, np.nan, dtype=np.float32)
    d[valid] = (calc[valid] - ctrl[valid]) / np.abs(ctrl[valid]) * 100.0
    stack[i] = d

# --- t-тест против нуля только внутри маски ---
print("t-тест против нуля по вокселям внутри маски...")
n_valid = np.sum(~np.isnan(stack), axis=0)
mean_d  = np.nanmean(stack, axis=0)
enough = (n_valid >= MIN_SUBJ) & gm
p_map = np.ones(shape, dtype=np.float32)
t_map = np.zeros(shape, dtype=np.float32)

idx = np.argwhere(enough)
print(f"  Вокселей для теста: {len(idx)}")
for (z,y,x) in idx:
    vals = stack[:, z, y, x]; vals = vals[~np.isnan(vals)]
    if len(vals) >= MIN_SUBJ and np.std(vals) > 0:
        t, p = stats.ttest_1samp(vals, 0.0)
        t_map[z,y,x] = t; p_map[z,y,x] = p

# --- FDR ---
flat_p = p_map[enough]
order = np.argsort(flat_p); m = len(flat_p)
ranked = flat_p[order]
thresh = FDR_Q * (np.arange(1, m+1) / m)
passed = ranked <= thresh
p_crit = ranked[np.max(np.where(passed)[0])] if passed.any() else 0.0
reliable = enough & (p_map <= p_crit) & (p_crit > 0)

# также посчитаем при простом p<0.05 и p<0.01 для сравнения
rel_05 = enough & (p_map < 0.05)
rel_01 = enough & (p_map < 0.01)

# --- сохранение ---
nib.save(nib.Nifti1Image(mean_d.astype(np.float32), affine), os.path.join(OUT,"delta_cmro2_mean.nii.gz"))
nib.save(nib.Nifti1Image(reliable.astype(np.uint8), affine), os.path.join(OUT,"reliable_mask.nii.gz"))
nib.save(nib.Nifti1Image(t_map, affine), os.path.join(OUT,"tstat.nii.gz"))

n_brain = int(enough.sum())
print(f"\n{'='*55}")
print(f"  Вокселей в маске (≥{MIN_SUBJ} субъектов): {n_brain}")
print(f"  Надёжных (FDR q<{FDR_Q}):  {int(reliable.sum())} ({100*reliable.sum()/n_brain:.1f}%)")
print(f"  При p<0.01 (без поправки): {int(rel_01.sum())} ({100*rel_01.sum()/n_brain:.1f}%)")
print(f"  При p<0.05 (без поправки): {int(rel_05.sum())} ({100*rel_05.sum()/n_brain:.1f}%)")
print(f"{'='*55}")
print(f"  Критики (тест значимости ΔCMRO2): ~22.8% надёжных")
print(f"  Сохранено в {OUT}/ (reliable_mask = FDR-вариант)")