import numpy as np, nibabel as nib
from scipy import stats
import os

BASE = "ds004873/derivatives"
GROUP_MASK = "ds004873/derivatives/masks/group_mask.nii.gz"
SUBJECTS = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055"]

gm = nib.load(GROUP_MASK).get_fdata() > 0
shape = gm.shape

# Загружаем BOLD PSC для gate
bold_stack = np.full((len(SUBJECTS),)+shape, np.nan, np.float32)
for i,s in enumerate(SUBJECTS):
    p = f"{BASE}/{s}/func/{s}_task-calccontrol_space-MNI152_res-2_desc-percchange_bold.nii.gz"
    if not os.path.exists(p): continue
    d = nib.load(p).get_fdata(dtype=np.float32)
    if d.shape != shape: continue
    bold_stack[i] = np.where(gm, d, np.nan)

enough = (np.sum(~np.isnan(bold_stack), axis=0) >= 15) & gm
res_bold = stats.ttest_1samp(bold_stack, 0.0, axis=0, nan_policy="omit")
p_bold = np.asarray(res_bold.pvalue); p_bold[~np.isfinite(p_bold)] = 1.0

# BH для gate
ps = np.sort(p_bold[enough]); m = len(ps)
thr = 0.05 * np.arange(1, m+1) / m
ok = ps <= thr
pcrit = ps[np.max(np.where(ok)[0])] if ok.any() else 0.0
bold_gate = enough & (p_bold <= pcrit) & (pcrit > 0)
print(f"BOLD-gate: {int(bold_gate.sum())} вокселей, pcrit={pcrit:.2e}")

# Загружаем semi-quant дельту внутри gate
stack = np.full((len(SUBJECTS),)+shape, np.nan, np.float32)
for i,s in enumerate(SUBJECTS):
    cp = f"{BASE}/{s}/qmri/{s}_task-calc_space-MNI152_desc-semi-quant-corrected_cmro2.nii.gz"
    kp = f"{BASE}/{s}/qmri/{s}_task-control_space-MNI152_desc-orig_cmro2.nii.gz"
    if not (os.path.exists(cp) and os.path.exists(kp)): continue
    c = nib.load(cp).get_fdata(dtype=np.float32)
    k = nib.load(kp).get_fdata(dtype=np.float32)
    v = np.isfinite(c)&np.isfinite(k)&(np.abs(k)>1e-3)&gm
    d = np.full(shape,np.nan,np.float32)
    d[v] = (c[v]-k[v])/np.abs(k[v])*100.0
    stack[i] = d

gate = bold_gate & (np.sum(~np.isnan(stack), axis=0) >= 15)
res = stats.ttest_1samp(stack, 0.0, axis=0, nan_policy="omit")
p = np.asarray(res.pvalue); p[~np.isfinite(p)] = 1.0
pin = p[gate]

print(f"\nСemi-quant ΔCMRO₂ внутри gate ({len(pin)} вокселей):")
print(f"  min p:      {pin.min():.2e}")
print(f"  p < 0.001:  {(pin<0.001).sum()} ({100*(pin<0.001).mean():.1f}%)")
print(f"  p < 0.01:   {(pin<0.01).sum()} ({100*(pin<0.01).mean():.1f}%)")
print(f"  p < 0.05:   {(pin<0.05).sum()} ({100*(pin<0.05).mean():.1f}%)")

# BH внутри gate
ps2 = np.sort(pin); m2 = len(ps2)
thr2 = 0.05 * np.arange(1, m2+1) / m2
ok2 = ps2 <= thr2
if ok2.any():
    pcrit2 = ps2[np.max(np.where(ok2)[0])]
    print(f"\n  FDR pcrit: {pcrit2:.2e}")
    print(f"  FDR проходят: {int((pin<=pcrit2).sum())} ({100*(pin<=pcrit2).mean():.1f}%)")
else:
    print(f"\n  FDR: ничего не прошло")
    print(f"  Самое маленькое p: {ps2[0]:.2e}")
    print(f"  Порог BH для него: {thr2[0]:.2e}")
    print(f"  Разрыв (во сколько раз p > порога): {ps2[0]/thr2[0]:.1f}x")




    import numpy as np
from scipy import stats

# Сколько вокселей прошло бы FDR если бы n=38 вместо 28?
# t при n=38 = t при n=28 × sqrt(38/28)
# p пересчитываем из новой t-статистики

# Берём реальные t-статистики из предыдущего прогона
# Проще: пересчитаем из p через ppf
n28 = 28; n38 = 38
# p → t(df=27) → масштабируем → p(df=37)
from scipy.stats import t as tdist
t28 = tdist.ppf(1 - pin/2, df=n28-1)          # двустороннее
t38 = t28 * np.sqrt(n38/n28)                   # масштабируем
p38 = 2 * (1 - tdist.cdf(np.abs(t38), df=n38-1))

ps38 = np.sort(p38); m = len(ps38)
thr = 0.05 * np.arange(1, m+1)/m
ok = ps38 <= thr
if ok.any():
    pcrit = ps38[np.max(np.where(ok)[0])]
    print(f"При n=38: FDR проходят {int((p38<=pcrit).sum())} ({100*(p38<=pcrit).mean():.1f}%)")
else:
    print(f"При n=38: FDR всё равно ничего не проходит")
    print(f"min p38 = {p38.min():.2e}, BH порог = {thr[0]:.2e}")