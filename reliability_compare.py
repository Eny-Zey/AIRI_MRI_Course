"""
Reliability-анализ с BOLD-gate — воспроизведение методологии Goltermann et al.
Stage 1: групповой t-test BOLD PSC -> FDR -> BOLD-gate
Stage 2: внутри gate, t-test ΔCMRO2≠0 -> FDR -> надёжные воксели
"""
import os
import numpy as np
import nibabel as nib
from scipy import stats

BASE = "ds004873/derivatives"
GROUP_MASK = "ds004873/derivatives/masks/group_mask.nii.gz"
FDR_Q = 0.05
MIN_SUBJ = 15
CTRL_MIN = 1e-3

SUBJECTS_OLD = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055"]

SUBJECTS_NEW = [
    "sub-p058","sub-p059","sub-p060","sub-p061","sub-p063",
    "sub-p064","sub-p065","sub-p066","sub-p068"]

SUBJECTS_ALL = SUBJECTS_OLD + SUBJECTS_NEW

# --- пути ---
bold_mni  = lambda s: f"{BASE}/{s}/func/{s}_task-calccontrol_space-MNI152_res-2_desc-percchange_bold.nii.gz"
calc_orig = lambda s: f"{BASE}/{s}/qmri/{s}_task-calc_space-MNI152_desc-orig_cmro2.nii.gz"
calc_semi = lambda s: f"{BASE}/{s}/qmri/{s}_task-calc_space-MNI152_desc-semi-quant-corrected_cmro2.nii.gz"
ctrl_path = lambda s: f"{BASE}/{s}/qmri/{s}_task-control_space-MNI152_desc-orig_cmro2.nii.gz"

gm = nib.load(GROUP_MASK).get_fdata() > 0
shape = gm.shape

def bh(p, q):
    ps = np.sort(p); m = len(ps)
    thr = q * np.arange(1, m+1) / m
    ok = ps <= thr
    if not ok.any(): return 0.0, 0, 1.0
    pcrit = ps[np.max(np.where(ok)[0])]
    n_sig = int((p <= pcrit).sum())
    return float(n_sig / m * 100), n_sig, pcrit

def build_bold_gate(subjects):
    """Строит BOLD-gate по заданному списку субъектов."""
    stack = np.full((len(subjects),) + shape, np.nan, np.float32)
    used = 0
    for i, s in enumerate(subjects):
        p = bold_mni(s)
        if not os.path.exists(p): continue
        d = nib.load(p).get_fdata(dtype=np.float32)
        if d.shape != shape: continue
        d[~np.isfinite(d)] = np.nan
        stack[i] = np.where(gm, d, np.nan)
        # добавь в build_bold_gate после stack[i] = np.where(gm, d, np.nan):
        if i == 0:
            n_valid = np.sum(~np.isnan(stack[0]))
            print(f"  Первый субъект: валидных вокселей в маске = {n_valid}")
        used += 1

    enough = (np.sum(~np.isnan(stack), axis=0) >= MIN_SUBJ) & gm
    with np.errstate(invalid="ignore"):
        res = stats.ttest_1samp(stack, 0.0, axis=0, nan_policy="omit")
    p = np.asarray(res.pvalue, dtype=np.float64)
    p[~np.isfinite(p)] = 1.0; p[~enough] = 1.0

    pin = p[enough]; m = int(enough.sum())
    _, _, pcrit = bh(pin, FDR_Q)
    gate = enough & (p <= pcrit) & (pcrit < 1.0)
    print(f"  субъектов={used}  вокселей с N>={MIN_SUBJ}: {m}")
    print(f"  BOLD-gate (FDR q<{FDR_Q}): {int(gate.sum())} ({100*gate.sum()/m:.1f}%)")
    return gate

def run_stage2(calc_fn, label, subjects, bold_gate):
    """t-test ΔCMRO2 внутри BOLD-gate для заданного списка субъектов."""
    stack = np.full((len(subjects),) + shape, np.nan, np.float32)
    used = 0
    for i, s in enumerate(subjects):
        cp, kp = calc_fn(s), ctrl_path(s)
        if not (os.path.exists(cp) and os.path.exists(kp)): continue
        c = nib.load(cp).get_fdata(dtype=np.float32)
        k = nib.load(kp).get_fdata(dtype=np.float32)
        if c.shape != shape or k.shape != shape: continue
        v = np.isfinite(c) & np.isfinite(k) & (np.abs(k) > CTRL_MIN) & gm
        d = np.full(shape, np.nan, np.float32)
        d[v] = (c[v] - k[v]) / np.abs(k[v]) * 100.0
        stack[i] = d; used += 1

    gate = bold_gate & (np.sum(~np.isnan(stack), axis=0) >= MIN_SUBJ)
    with np.errstate(invalid="ignore"):
        res = stats.ttest_1samp(stack, 0.0, axis=0, nan_policy="omit")
    p = np.asarray(res.pvalue, dtype=np.float64)
    p[~np.isfinite(p)] = 1.0
    pin = p[gate]; n = int(gate.sum())

    pct_fdr, n_fdr, _ = bh(pin, FDR_Q)
    print(f"  [{label}]  субъектов={used}  вокселей в gate={n}")
    print(f"    FDR q<{FDR_Q}:     {pct_fdr:.1f}%  ({n_fdr} вокселей)")
    print(f"    p<0.01 без попр.: {100*np.mean(pin<0.01):.1f}%")
    print(f"    p<0.05 без попр.: {100*np.mean(pin<0.05):.1f}%")

# ================================================================
# ПРОГОН 1: все субъекты (n=38) — основной результат
# ================================================================
print("=" * 60)
print(f"ПРОГОН 1: все субъекты (n={len(SUBJECTS_ALL)})")
print("=" * 60)
print("\n--- Stage 1: BOLD-gate ---")
gate_all = build_bold_gate(SUBJECTS_ALL)
print("\n--- Stage 2: ΔCMRO2 ---")
run_stage2(calc_orig, "ORIG calc",      SUBJECTS_ALL, gate_all)
run_stage2(calc_semi, "SEMI-QUANT calc", SUBJECTS_ALL, gate_all)

# ================================================================
# ПРОГОН 2: только старые субъекты (n=29)
# ================================================================
print("\n" + "=" * 60)
print(f"ПРОГОН 2: только старые субъекты (n={len(SUBJECTS_OLD)})")
print("=" * 60)
print("\n--- Stage 1: BOLD-gate ---")
gate_old = build_bold_gate(SUBJECTS_OLD)
print("\n--- Stage 2: ΔCMRO2 ---")
run_stage2(calc_orig, "ORIG calc",      SUBJECTS_OLD, gate_old)
run_stage2(calc_semi, "SEMI-QUANT calc", SUBJECTS_OLD, gate_old)

# ================================================================
# ПРОГОН 3: только новые субъекты (n=9)
# ================================================================
print("\n" + "=" * 60)
print(f"ПРОГОН 3: только новые субъекты (n={len(SUBJECTS_NEW)})")
print("=" * 60)
print("\n--- Stage 1: BOLD-gate ---")
gate_new = build_bold_gate(SUBJECTS_NEW)
print("\n--- Stage 2: ΔCMRO2 ---")
run_stage2(calc_orig, "ORIG calc",      SUBJECTS_NEW, gate_new)
run_stage2(calc_semi, "SEMI-QUANT calc", SUBJECTS_NEW, gate_new)

print("\n" + "=" * 60)
print("Бенчмарк критиков (semi-quant, FDR внутри BOLD-gate, n=38): ~22.8%")
print("=" * 60)