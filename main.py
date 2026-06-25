import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img


ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026",
    "sub-p027", "sub-p028", "sub-p030", "sub-p031", "sub-p032",
    "sub-p033", "sub-p034", "sub-p035", "sub-p036", "sub-p037",
    "sub-p038", "sub-p039", "sub-p040", "sub-p043", "sub-p044",
    "sub-p046", "sub-p047", "sub-p048", "sub-p049", "sub-p050",
    "sub-p051", "sub-p052", "sub-p054", "sub-p055",
]


SUB_ID = "sub-p026"

BASE_PATH = "./ds004873"
OUTPUT_DIR = "./labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Пороги
Z_THRESH = 2.3              # p < 0.01, порог значимости BOLD Z-статистики
CMRO2_MIN_PCT_CHANGE = 1.0  # минимальное значимое ΔCMRO₂ в %
                            # статья: ~3% в CALC-positive вокселях — 1% консервативный нижний порог
CMRO2_MIN_CTRL = 80.0  # физиологический минимум CMRO₂ (µmol/100g/min)

# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------------------
t1_img = nib.load(os.path.join(BASE_PATH, "derivatives", SUB_ID, "anat",
    f"{SUB_ID}_desc-fmriprep_T1w.nii.gz"))
t1_data = t1_img.get_fdata().astype(np.float32)

contrast_img = nib.load(os.path.join(
    BASE_PATH, "derivatives", SUB_ID, "func",
    f"{SUB_ID}_1stlevel_calccontrol_space-T2.nii.gz"
))

calc_cmro2_img = nib.load(os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
    f"{SUB_ID}_task-calc_space-T1w_desc-orig_cmro2.nii.gz"))
ctrl_cmro2_img = nib.load(os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
    f"{SUB_ID}_task-control_space-T1w_desc-orig_cmro2.nii.gz"))

# CBF загружен, но не используется в правиле классификации.
# По статье (Epp et al., 2025) конкордантность/дискордантность определяется
# только по совпадению знаков ΔBOLD и ΔCMRO₂.
# CBF важен для характеристики механизма (конкордантные воксели регулируют
# CMRO₂ через ΔCBF, дискордантные — через ΔOEF), но не для метки.
# Оставлен для возможного расширения анализа.
# CBF — загружаем если есть, не критично для классификации
calc_cbf_img = None
ctrl_cbf_img = None

calc_cbf_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
    f"{SUB_ID}_task-calc_space-T1w_cbf.nii.gz")
ctrl_cbf_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
    f"{SUB_ID}_task-control_space-T1w_cbf.nii.gz")

if os.path.exists(calc_cbf_path) and os.path.exists(ctrl_cbf_path):
    calc_cbf_img = nib.load(calc_cbf_path)
    ctrl_cbf_img = nib.load(ctrl_cbf_path)
    print(f"  CBF карты загружены")
else:
    print(f"  CBF карты не найдены, пропускаем (не влияет на разметку)")
brain_mask_img = nib.load(os.path.join(
    BASE_PATH, "derivatives", SUB_ID, "anat",
    f"{SUB_ID}_desc-fmriprep_brain_mask.nii.gz"
))

csf_mask_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "anat",
    f"{SUB_ID}_BrMsk_CSF.nii")

# ----------------------------------------------------------------------
# 2. РЕСЕМПЛИНГ ВСЕГО В ПРОСТРАНСТВО T1w
# ----------------------------------------------------------------------
def resample_if_needed(img, ref_img, interpolation='linear'):
    """Ресемплирует img к ref_img только если размеры не совпадают."""
    if img.shape[:3] != ref_img.shape[:3]:
        fname = getattr(img, 'get_filename', lambda: '?')()
        print(f"  Ресемплинг {fname} -> {ref_img.shape[:3]}")
        return resample_to_img(img, ref_img, interpolation=interpolation)
    return img

contrast_r_img   = resample_if_needed(contrast_img,   t1_img, 'linear')
calc_cmro2_r_img = resample_if_needed(calc_cmro2_img, t1_img, 'linear')
ctrl_cmro2_r_img = resample_if_needed(ctrl_cmro2_img, t1_img, 'linear')
brain_mask_r_img = resample_if_needed(brain_mask_img, t1_img, 'nearest')

out_contrast = os.path.join(OUTPUT_DIR, f"{SUB_ID}_calccontrol_space-T1w.nii.gz")
nib.save(contrast_r_img, out_contrast)
print(f"  Сохранён: {out_contrast}")

contrast_t1w = contrast_r_img.get_fdata().astype(np.float32)
calc_cmro2   = calc_cmro2_r_img.get_fdata().astype(np.float32)
ctrl_cmro2   = ctrl_cmro2_r_img.get_fdata().astype(np.float32)
brain_mask   = brain_mask_r_img.get_fdata().astype(bool)

# ----------------------------------------------------------------------
# 3. МАСКА СЕРОГО ВЕЩЕСТВА
#    Статья анализирует только gray matter: CMRO₂ в белом веществе
#    не интерпретируется из-за эффектов миелинизации.
#    fmriprep обычно сохраняет карту вероятности как _label-GM_probseg.nii.gz
# ----------------------------------------------------------------------
gm_mask_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "anat",
    f"{SUB_ID}_label-GM_probseg.nii.gz")

if os.path.exists(gm_mask_path):
    gm_r_img = resample_if_needed(nib.load(gm_mask_path), t1_img, 'linear')
    gm_mask = gm_r_img.get_fdata().astype(np.float32) > 0.5
    combined_mask = brain_mask & gm_mask
    print(f"  Маска GM применена (вокселей в GM: {np.sum(gm_mask)})")
else:
    combined_mask = brain_mask  # CSF отфильтруется порогом CMRO2_MIN_CTRL = 20
    print(f"  GM маска не найдена, используется brain_mask + порог CMRO2 > {CMRO2_MIN_CTRL}")
# ----------------------------------------------------------------------
# 4. ОТНОСИТЕЛЬНОЕ ИЗМЕНЕНИЕ CMRO₂ (в %)
#    Статья везде использует ΔCMRO₂[%], а не абсолютные единицы.
#    Базовое CMRO₂ в сером веществе ~130-170 µmol/100g/min,
#    поэтому порог 0.01 в абсолютных единицах был фактически нулём.
#    eps защищает от деления на нуль и отфильтровывает воксели
#    с физически нереалистичным базовым CMRO₂ (≤ 0).
# ----------------------------------------------------------------------
eps = 1e-6
delta_cmro2_pct = (calc_cmro2 - ctrl_cmro2) / (np.abs(ctrl_cmro2) + eps) * 100.0

# ----------------------------------------------------------------------
# 5. МАСКА АКТИВНЫХ ВОКСЕЛЕЙ
# ----------------------------------------------------------------------
mask_active = (np.abs(contrast_t1w) > Z_THRESH) & combined_mask
print(f"  Активных вокселей (|Z| > {Z_THRESH}): {np.sum(mask_active)}")

# ----------------------------------------------------------------------
# 6. КЛАССИФИКАЦИЯ (векторизованная)
#    Правило из Epp et al. 2025:
#      Конкордантный: sign(ΔBOLD) == sign(ΔCMRO₂[%])  → метка 1
#      Дискордантный: sign(ΔBOLD) != sign(ΔCMRO₂[%])  → метка 2
# ----------------------------------------------------------------------
valid_mask = (
    mask_active
    & ~np.isnan(contrast_t1w)    & ~np.isinf(contrast_t1w)
    & ~np.isnan(delta_cmro2_pct) & ~np.isinf(delta_cmro2_pct)
    & (ctrl_cmro2 > CMRO2_MIN_CTRL)
    & (np.abs(delta_cmro2_pct) >= CMRO2_MIN_PCT_CHANGE)
)

concordant_mask = valid_mask & (np.sign(contrast_t1w) == np.sign(delta_cmro2_pct))
discordant_mask = valid_mask & (np.sign(contrast_t1w) != np.sign(delta_cmro2_pct))

label_map = np.zeros_like(t1_data, dtype=np.uint8)
label_map[concordant_mask] = 1
label_map[discordant_mask] = 2

num_concordant = int(np.sum(concordant_mask))
num_discordant = int(np.sum(discordant_mask))
total = num_concordant + num_discordant

print(f"\n  Конкордантных:  {num_concordant}")
print(f"  Дискордантных:  {num_discordant}")
print(f"  Всего прошли фильтрацию: {total}")
if total > 0:
    pct = num_discordant / total * 100
    print(f"  Доля дискордантных: {pct:.1f}%")
    print(f"  (Ожидаемо по статье: ~31-40% среди positive BOLD,")
    print(f"                       ~52-66% среди negative BOLD)")

# ----------------------------------------------------------------------
# 7. СОХРАНЕНИЕ
# ----------------------------------------------------------------------
label_header = t1_img.header.copy()
label_header.set_data_dtype(np.uint8)
label_header['scl_slope'] = 1.0
label_header['scl_inter'] = 0.0
label_img = nib.Nifti1Image(label_map, affine=t1_img.affine, header=label_header)

label_path = os.path.join(OUTPUT_DIR, f"{SUB_ID}_label-concordance.nii.gz")
nib.save(label_img, label_path)
print(f"\n  Разметка сохранена: {label_path}")

# ΔCMRO₂[%] — полезно открыть в MRIcron рядом с разметкой
delta_cmro2_clipped = np.where(combined_mask, delta_cmro2_pct, 0)
delta_cmro2_clipped = np.clip(delta_cmro2_clipped, -20.0, 20.0)  # ±20% — разумный диапазон

delta_header = t1_img.header.copy()
delta_header.set_data_dtype(np.float32)
delta_header['scl_slope'] = 1.0
delta_header['scl_inter'] = 0.0

delta_cmro2_img = nib.Nifti1Image(
    delta_cmro2_clipped.astype(np.float32),
    affine=t1_img.affine,
    header=delta_header
)
nib.save(delta_cmro2_img, os.path.join(OUTPUT_DIR, f"{SUB_ID}_delta_cmro2_pct.nii.gz"))
print(f"  ΔCMRO₂[%] сохранён для визуализации")

