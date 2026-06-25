import os
import csv
import hashlib
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

# ----------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ----------------------------------------------------------------------
BASE_PATH = "./ds004873"
OUTPUT_DIR = "./labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_THRESH          = 1.96   # p < 0.05, порог значимости BOLD Z-статистики
CMRO2_MIN_PCT_CHANGE = 3.0 # минимальное значимое ΔCMRO₂ в %
CMRO2_MIN_CTRL    = 80.0   # временный порог вместо GM-маски (µmol/100g/min)

ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026",
    "sub-p027", "sub-p028", "sub-p030", "sub-p031", "sub-p032",
    "sub-p033", "sub-p034", "sub-p035", "sub-p036", "sub-p037",
    "sub-p038", "sub-p039", "sub-p040", "sub-p043", "sub-p044",
    "sub-p046", "sub-p047", "sub-p048", "sub-p049", "sub-p050",
    "sub-p051", "sub-p052", "sub-p054", "sub-p055",
]

# ----------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------------------------------------------------
def resample_if_needed(img, ref_img, interpolation='linear'):
    if img.shape[:3] != ref_img.shape[:3]:
        return resample_to_img(img, ref_img, interpolation=interpolation)
    return img

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def files_are_identical(path_a, path_b):
    """Проверяет идентичность двух файлов по MD5."""
    return md5(path_a) == md5(path_b)

# ----------------------------------------------------------------------
# ФУНКЦИЯ РАЗМЕТКИ ОДНОГО СУБЪЕКТА
# ----------------------------------------------------------------------
def label_subject(sub_id):
    """
    Возвращает словарь с результатами или None если субъект пропущен.
    """
    print(f"\n{'='*60}")
    print(f"  {sub_id}")
    print(f"{'='*60}")

    # --- Пути к файлам ---
    t1_path       = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                        f"{sub_id}_desc-fmriprep_T1w.nii.gz")
    contrast_path = os.path.join(BASE_PATH, "derivatives", sub_id, "func",
                        f"{sub_id}_1stlevel_calccontrol_space-T2.nii.gz")
    calc_cmro2_path = os.path.join(BASE_PATH, "derivatives", sub_id, "qmri",
                        f"{sub_id}_task-calc_space-T1w_desc-orig_cmro2.nii.gz")
    ctrl_cmro2_path = os.path.join(BASE_PATH, "derivatives", sub_id, "qmri",
                        f"{sub_id}_task-control_space-T1w_desc-orig_cmro2.nii.gz")
    brain_mask_path = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                        f"{sub_id}_desc-fmriprep_brain_mask.nii.gz")
    gm_mask_path  = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                        f"{sub_id}_label-GM_probseg.nii.gz")
    calc_cbf_path = os.path.join(BASE_PATH, "derivatives", sub_id, "qmri",
                        f"{sub_id}_task-calc_space-T1w_cbf.nii.gz")
    ctrl_cbf_path = os.path.join(BASE_PATH, "derivatives", sub_id, "qmri",
                        f"{sub_id}_task-control_space-T1w_cbf.nii.gz")

    # --- Проверка обязательных файлов ---
    required = {
        "T1w":        t1_path,
        "BOLD":       contrast_path,
        "calc_cmro2": calc_cmro2_path,
        "ctrl_cmro2": ctrl_cmro2_path,
        "brain_mask": brain_mask_path,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print(f"  ✗ ПРОПУЩЕН — нет файлов: {', '.join(missing)}")
        return {"sub_id": sub_id, "status": f"missing: {','.join(missing)}",
                "concordant": 0, "discordant": 0, "total": 0, "pct_discordant": None}

    # --- Проверка на идентичность calc/control cmro2 (как у sub-p029) ---
    if files_are_identical(calc_cmro2_path, ctrl_cmro2_path):
        print(f"  ✗ ПРОПУЩЕН — calc_cmro2 и ctrl_cmro2 идентичны (нет delta)")
        return {"sub_id": sub_id, "status": "identical_cmro2",
                "concordant": 0, "discordant": 0, "total": 0, "pct_discordant": None}

    # --- Загрузка ---
    try:
        t1_img          = nib.load(t1_path)
        t1_data         = t1_img.get_fdata().astype(np.float32)
        contrast_img    = nib.load(contrast_path)
        calc_cmro2_img  = nib.load(calc_cmro2_path)
        ctrl_cmro2_img  = nib.load(ctrl_cmro2_path)
        brain_mask_img  = nib.load(brain_mask_path)
    except Exception as e:
        print(f"  ✗ ПРОПУЩЕН — ошибка загрузки: {e}")
        return {"sub_id": sub_id, "status": f"load_error: {e}",
                "concordant": 0, "discordant": 0, "total": 0, "pct_discordant": None}

    if os.path.exists(calc_cbf_path) and os.path.exists(ctrl_cbf_path):
        print(f"  CBF карты загружены")
    else:
        print(f"  CBF карты не найдены, пропускаем")

    # --- Ресемплинг в пространство T1w ---
    contrast_r_img   = resample_if_needed(contrast_img,   t1_img, 'linear')
    calc_cmro2_r_img = resample_if_needed(calc_cmro2_img, t1_img, 'linear')
    ctrl_cmro2_r_img = resample_if_needed(ctrl_cmro2_img, t1_img, 'linear')
    brain_mask_r_img = resample_if_needed(brain_mask_img, t1_img, 'nearest')

    # Сохраняем ресемплированный BOLD для визуального контроля
    out_contrast = os.path.join(OUTPUT_DIR, f"{sub_id}_calccontrol_space-T1w.nii.gz")
    nib.save(contrast_r_img, out_contrast)

    contrast_t1w = contrast_r_img.get_fdata().astype(np.float32)
    calc_cmro2   = calc_cmro2_r_img.get_fdata().astype(np.float32)
    ctrl_cmro2   = ctrl_cmro2_r_img.get_fdata().astype(np.float32)
    brain_mask   = brain_mask_r_img.get_fdata().astype(bool)

    # --- Маска серого вещества (если есть) ---
    if os.path.exists(gm_mask_path):
        gm_r_img = resample_if_needed(nib.load(gm_mask_path), t1_img, 'linear')
        gm_mask = gm_r_img.get_fdata().astype(np.float32) > 0.5
        combined_mask = brain_mask & gm_mask
        print(f"  GM маска применена (вокселей: {np.sum(gm_mask)})")
    else:
        combined_mask = brain_mask
        print(f"  GM маска не найдена, brain_mask + CMRO2 > {CMRO2_MIN_CTRL}")

    # --- ΔCMRO₂ в % ---
    eps = 1e-6
    delta_cmro2_pct = (calc_cmro2 - ctrl_cmro2) / (np.abs(ctrl_cmro2) + eps) * 100.0

    # --- Маска активных вокселей ---
    mask_active = (np.abs(contrast_t1w) > Z_THRESH) & combined_mask
    print(f"  Активных вокселей (|Z| > {Z_THRESH}): {np.sum(mask_active)}")

    # --- Классификация ---
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
    pct = num_discordant / total * 100 if total > 0 else None

    print(f"  Конкордантных:  {num_concordant}")
    print(f"  Дискордантных:  {num_discordant}")
    print(f"  Всего: {total}")
    if pct is not None:
        print(f"  Доля дискордантных: {pct:.1f}%")

    # Предупреждение если разметка пустая
    if total == 0:
        print(f"  ⚠️  ВНИМАНИЕ: 0 вокселей прошли фильтрацию — проверь данные!")

    # --- Сохранение разметки ---
    label_header = t1_img.header.copy()
    label_header.set_data_dtype(np.uint8)
    label_header['scl_slope'] = 1.0
    label_header['scl_inter'] = 0.0
    label_img = nib.Nifti1Image(label_map, affine=t1_img.affine, header=label_header)
    label_path = os.path.join(OUTPUT_DIR, f"{sub_id}_label-concordance.nii.gz")
    nib.save(label_img, label_path)
    print(f"  Разметка сохранена: {label_path}")

    return {
        "sub_id":         sub_id,
        "status":         "ok" if total > 0 else "empty",
        "concordant":     num_concordant,
        "discordant":     num_discordant,
        "total":          total,
        "pct_discordant": round(pct, 1) if pct is not None else None,
    }

# ----------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ
# ----------------------------------------------------------------------
if __name__ == "__main__":
    results = []

    for sub_id in ALL_SUBJECTS:
        result = label_subject(sub_id)
        results.append(result)

    # --- Сводная таблица ---
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'='*60}")
    print(f"{'Субъект':<15} {'Статус':<20} {'Конкорд.':<12} {'Дискорд.':<12} {'% дискорд.'}")
    print("-" * 65)

    ok_results = []
    for r in results:
        pct_str = f"{r['pct_discordant']:.1f}%" if r['pct_discordant'] is not None else "—"
        print(f"{r['sub_id']:<15} {r['status']:<20} {r['concordant']:<12} {r['discordant']:<12} {pct_str}")
        if r['status'] == 'ok':
            ok_results.append(r)

    if ok_results:
        pcts = [r['pct_discordant'] for r in ok_results]
        totals = [r['total'] for r in ok_results]
        print("-" * 65)
        print(f"  Успешно размечено субъектов: {len(ok_results)} / {len(ALL_SUBJECTS)}")
        print(f"  Медиана % дискордантных: {np.median(pcts):.1f}%")
        print(f"  Среднее % дискордантных: {np.mean(pcts):.1f}%")
        print(f"  Медиана вокселей на субъекта: {int(np.median(totals))}")
        print(f"  Всего вокселей: {sum(totals)}")
        print(f"  (Ожидаемо по статье: медиана ~31-40% у positive BOLD)")

    # --- Сохранение CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "labeling_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sub_id", "status", "concordant", "discordant", "total", "pct_discordant"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV сохранён: {csv_path}")
