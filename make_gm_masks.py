import os
import ants
import numpy as np

# ----------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ----------------------------------------------------------------------
BASE_PATH = "./ds004873"

GM_PROB_THRESHOLD = 0.5   # порог бинаризации вероятностной карты GM


ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026",
    "sub-p027", "sub-p028", "sub-p030", "sub-p031", "sub-p032",
    "sub-p033", "sub-p034", "sub-p035", "sub-p036", "sub-p037",
    "sub-p038", "sub-p039", "sub-p040", "sub-p043", "sub-p044",
    "sub-p046", "sub-p047", "sub-p048", "sub-p049", "sub-p050",
    "sub-p051", "sub-p052", "sub-p054", "sub-p055",
]

# ----------------------------------------------------------------------
# СЕГМЕНТАЦИЯ ОДНОГО СУБЪЕКТА
# ----------------------------------------------------------------------
def segment_subject(sub_id):
    anat_dir = os.path.join(BASE_PATH, "derivatives", sub_id, "anat")
    brain_path = os.path.join(anat_dir, f"{sub_id}_desc-fmriprep_T1w_brain.nii.gz")
    out_path   = os.path.join(anat_dir, f"{sub_id}_label-GM_probseg.nii.gz")

    if os.path.exists(out_path):
        print(f"  ✓ {sub_id}: GM-маска уже есть, пропускаю")
        return "skip"

    if not os.path.exists(brain_path):
        print(f"  ✗ {sub_id}: нет {brain_path}")
        return "missing"

    print(f"  {sub_id}: сегментация...")

    # Загружаем извлечённый мозг
    img = ants.image_read(brain_path)

    # Маска мозга = всё что не ноль
    mask = ants.get_mask(img)

    # Atropos: 3 ткани (1=CSF, 2=GM, 3=WM)
    # i='kmeans[3]' — инициализация k-means на 3 класса
    seg = ants.atropos(
        a=img,
        x=mask,
        i='kmeans[3]',
        m='[0.2,1x1x1]',   # MRF сглаживание
        c='[5,0]'          # 5 итераций
    )

    # seg['probabilityimages'] — список из 3 вероятностных карт
    # Класс с промежуточной средней интенсивностью = GM
    # (CSF тёмный, WM яркий, GM между ними)
    # Определяем порядок классов по средней интенсивности T1w
    img_np = img.numpy()
    means = []
    for k in range(3):
        prob_k = seg['probabilityimages'][k].numpy()
        # средняя интенсивность вокселей, отнесённых к классу k
        cls_voxels = img_np[prob_k > 0.5]
        means.append(cls_voxels.mean() if len(cls_voxels) > 0 else 0)

    # GM — средний по интенсивности класс (не самый тёмный, не самый яркий)
    gm_class = int(np.argsort(means)[1])  # средний из трёх
    print(f"    Средние интенсивности классов: {[f'{m:.0f}' for m in means]}, "
          f"GM = класс {gm_class}")

    gm_prob = seg['probabilityimages'][gm_class]
    gm_mask = ants.threshold_image(gm_prob, GM_PROB_THRESHOLD, 1.0, 1, 0)

    # Сохраняем
    ants.image_write(gm_mask, out_path)
    n_gm = int(gm_mask.numpy().sum())
    print(f"    Сохранено: {out_path} (GM-вокселей: {n_gm})")
    return "ok"

# ----------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ
# ----------------------------------------------------------------------
if __name__ == "__main__":
    stats = {"ok": 0, "skip": 0, "missing": 0, "error": 0}

    for sub_id in ALL_SUBJECTS:
        print(f"\n{'='*50}\n  {sub_id}\n{'='*50}")
        try:
            result = segment_subject(sub_id)
            stats[result] += 1
        except Exception as e:
            print(f"  ✗ {sub_id}: ОШИБКА — {e}")
            stats["error"] += 1

    print(f"\n{'='*50}\nИТОГ\n{'='*50}")
    print(f"  Создано: {stats['ok']}")
    print(f"  Пропущено (уже были): {stats['skip']}")
    print(f"  Нет файлов: {stats['missing']}")
    print(f"  Ошибок: {stats['error']}")
