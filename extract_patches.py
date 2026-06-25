import os
import numpy as np
import nibabel as nib

# ----------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ----------------------------------------------------------------------
BASE_PATH   = "./ds004873"
LABELS_DIR  = "./labels"
OUTPUT_DIR  = "./patches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE        = 32     # размер куба: 16 или 32
PATCHES_PER_CLASS = 800   # на класс на субъекта (2500*2 = 5000 на субъекта)
RANDOM_SEED       = 42

ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026",
    "sub-p027", "sub-p028", "sub-p030", "sub-p031", "sub-p032",
    "sub-p033", "sub-p034", "sub-p035", "sub-p036", "sub-p037",
    "sub-p038", "sub-p039", "sub-p040", "sub-p043", "sub-p044",
    "sub-p046", "sub-p047", "sub-p048", "sub-p049", "sub-p050",
    "sub-p051", "sub-p052", "sub-p054", "sub-p055",
]

# ----------------------------------------------------------------------
# ПРОХОД 1: СБОР КООРДИНАТ (лёгкий, без самих патчей)
# ----------------------------------------------------------------------
def collect_coords(rng):
    """Возвращает список (sub_id, t1_path, center_tuple, label01)."""
    plan = []
    half = PATCH_SIZE // 2

    for sub_id in ALL_SUBJECTS:
        t1_path    = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                        f"{sub_id}_desc-fmriprep_T1w.nii.gz")
        label_path = os.path.join(LABELS_DIR, f"{sub_id}_label-concordance.nii.gz")
        if not os.path.exists(t1_path) or not os.path.exists(label_path):
            print(f"  x {sub_id}: нет файлов - пропускаю")
            continue

        labels = nib.load(label_path).get_fdata().astype(np.uint8)
        shape = labels.shape

        for cls in (1, 2):  # 1=конкорд., 2=дискорд.
            coords = np.argwhere(labels == cls)
            if len(coords) == 0:
                print(f"  ! {sub_id}: нет вокселей класса {cls}")
                continue
            # отбрасываем приграничные (патч должен помещаться целиком)
            valid = coords[
                (coords[:, 0] >= half) & (coords[:, 0] <= shape[0] - half) &
                (coords[:, 1] >= half) & (coords[:, 1] <= shape[1] - half) &
                (coords[:, 2] >= half) & (coords[:, 2] <= shape[2] - half)
            ]
            if len(valid) == 0:
                continue
            rng.shuffle(valid)
            take = valid[:PATCHES_PER_CLASS]
            label01 = 0 if cls == 1 else 1
            for c in take:
                plan.append((sub_id, t1_path, tuple(int(v) for v in c), label01))
            print(f"  {sub_id} класс {cls}: запланировано {len(take)} "
                  f"(валидных {len(valid)})")

    return plan

# ----------------------------------------------------------------------
# ПРОХОД 2: ИЗВЛЕЧЕНИЕ В ПРЕДВЫДЕЛЕННЫЙ МАССИВ
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    print("ПРОХОД 1/2: сбор координат...")
    plan = collect_coords(rng)
    N = len(plan)
    print(f"\n  Всего патчей: {N}")

    # Предвыделяем массивы СРАЗУ нужного размера (без растущих списков!)
    X = np.empty((N, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    y = np.empty(N, dtype=np.uint8)
    subj = np.empty(N, dtype=object)

    print("\nПРОХОД 2/2: извлечение патчей...")
    # сортируем по субъекту - T1w грузим один раз на субъекта
    plan.sort(key=lambda r: r[0])

    half = PATCH_SIZE // 2
    cur_sub = None
    t1 = None
    for i, (sub_id, t1_path, center, label01) in enumerate(plan):
        if sub_id != cur_sub:
            t1 = nib.load(t1_path).get_fdata().astype(np.float32)
            cur_sub = sub_id
            print(f"  {sub_id}...")
        z, yy, x = center
        X[i, 0] = t1[z-half:z+half, yy-half:yy+half, x-half:x+half]
        y[i] = label01
        subj[i] = sub_id

    # Перемешиваем единым образом (чтобы классы/субъекты не шли блоками)
    perm = rng.permutation(N)
    X, y, subj = X[perm], y[perm], subj[perm]

    print(f"\n{'='*50}\nИТОГ\n{'='*50}")
    print(f"  Форма X: {X.shape}")
    print(f"  Конкордантных (0): {int(np.sum(y == 0))}")
    print(f"  Дискордантных (1): {int(np.sum(y == 1))}")
    print(f"  Субъектов: {len(np.unique(subj))}")
    print(f"  В памяти: {X.nbytes / 1e9:.2f} ГБ")

    # Сохраняем в сжатый npz (один файл - удобно залить в Colab/Drive)
    out_path = os.path.join(OUTPUT_DIR, f"patches_{PATCH_SIZE}.npz")
    print(f"\n  Сохранение в {out_path} (сжатие может занять минуту)...")
    np.savez_compressed(out_path,
                        X=X, y=y, subj=subj.astype(str),
                        patch_size=PATCH_SIZE)
    print(f"  Готово. Размер на диске: {os.path.getsize(out_path) / 1e9:.2f} ГБ")

    # Сводка по субъектам: сколько патчей у каждого
    print(f"\n  Патчей на субъекта:")
    uniq, counts = np.unique(subj, return_counts=True)
    for s, c in zip(uniq, counts):
        flag = "  ! мало" if c < 2 * PATCHES_PER_CLASS else ""
        print(f"    {s}: {c}{flag}")