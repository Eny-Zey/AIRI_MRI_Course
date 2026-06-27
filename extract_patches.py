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

PATCH_SIZE        = 32     # размер куба
PATCHES_PER_CLASS = 1500   # на класс на субъекта
RANDOM_SEED       = 42

# --- ФИЛЬТР ПО ЧИСТОТЕ РАЗМЕТКИ ---
PURITY_FILTER = True
PURITY_THRESH = 0.8    # порог чистоты окрестности
PURITY_WIN    = 16     # окно оценки чистоты (меньше патча - это ОК)

ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026",
    "sub-p027", "sub-p028", "sub-p030", "sub-p031", "sub-p032",
    "sub-p033", "sub-p034", "sub-p035", "sub-p036", "sub-p037",
    "sub-p038", "sub-p039", "sub-p040", "sub-p043", "sub-p044",
    "sub-p046", "sub-p047", "sub-p048", "sub-p049", "sub-p050",
    "sub-p051", "sub-p052", "sub-p054", "sub-p055",
]

# ----------------------------------------------------------------------
# ЧИСТОТА ОКРЕСТНОСТИ
# ----------------------------------------------------------------------
def neighborhood_purity(labels, z, y, x, win):
    h = win // 2
    cube = labels[z-h:z+h, y-h:y+h, x-h:x+h]
    labeled = cube[cube > 0]
    if len(labeled) == 0:
        return 0.0
    return np.sum(labeled == labels[z, y, x]) / len(labeled)

# ----------------------------------------------------------------------
# ПРОХОД 1: СБОР КООРДИНАТ С ФИЛЬТРОМ ЧИСТОТЫ
# ----------------------------------------------------------------------
def collect_coords(rng):
    plan = []
    half = PATCH_SIZE // 2
    hp   = max(half, PURITY_WIN // 2)

    for sub_id in ALL_SUBJECTS:
        t1_path    = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                        f"{sub_id}_desc-fmriprep_T1w.nii.gz")
        label_path = os.path.join(LABELS_DIR, f"{sub_id}_label-concordance.nii.gz")
        if not os.path.exists(t1_path) or not os.path.exists(label_path):
            print(f"  x {sub_id}: нет файлов - пропускаю")
            continue

        labels = nib.load(label_path).get_fdata().astype(np.uint8)
        shape = labels.shape

        for cls in (1, 2):
            coords = np.argwhere(labels == cls)
            if len(coords) == 0:
                continue
            valid = coords[
                (coords[:,0] >= hp) & (coords[:,0] <= shape[0]-hp) &
                (coords[:,1] >= hp) & (coords[:,1] <= shape[1]-hp) &
                (coords[:,2] >= hp) & (coords[:,2] <= shape[2]-hp)
            ]
            if len(valid) == 0:
                continue
            rng.shuffle(valid)

            label01 = 0 if cls == 1 else 1
            collected = 0
            for c in valid:
                if collected >= PATCHES_PER_CLASS:
                    break
                z, y, x = int(c[0]), int(c[1]), int(c[2])
                if PURITY_FILTER:
                    if neighborhood_purity(labels, z, y, x, PURITY_WIN) < PURITY_THRESH:
                        continue
                plan.append((sub_id, t1_path, (z, y, x), label01))
                collected += 1
            print(f"  {sub_id} класс {cls}: собрано {collected}")

    return plan

# ----------------------------------------------------------------------
# ПРОХОД 2: ИЗВЛЕЧЕНИЕ С ПОТОКОВОЙ ЗАПИСЬЮ В MEMMAP (не копит в RAM)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    print("ПРОХОД 1/2: сбор координат с фильтром чистоты...")
    if PURITY_FILTER:
        print(f"  Фильтр: чистота окна {PURITY_WIN}^3 >= {PURITY_THRESH}\n")
    plan = collect_coords(rng)
    N = len(plan)
    print(f"\n  Всего патчей: {N}")

    # Перемешиваем ПОРЯДОК записей заранее (чтобы классы/субъекты не блоками),
    # но сортировку по субъекту делаем для эффективной загрузки T1w.
    # Решение: пишем в memmap по субъектам, потом сохраняем перемешанный индекс.

    suffix = "_pure" if PURITY_FILTER else ""
    X_path = os.path.join(OUTPUT_DIR, f"X_{PATCH_SIZE}{suffix}.npy")
    y_path = os.path.join(OUTPUT_DIR, f"y_{PATCH_SIZE}{suffix}.npy")
    s_path = os.path.join(OUTPUT_DIR, f"subj_{PATCH_SIZE}{suffix}.npy")

    # Создаём memmap на диске (НЕ в RAM)
    X_mm = np.lib.format.open_memmap(
        X_path, mode='w+', dtype=np.float32,
        shape=(N, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
    y_arr = np.empty(N, dtype=np.int64)
    s_arr = np.empty(N, dtype=object)

    print("\nПРОХОД 2/2: извлечение (потоковая запись на диск)...")
    plan.sort(key=lambda r: r[0])   # по субъекту - T1w грузим один раз

    half = PATCH_SIZE // 2
    cur_sub, t1 = None, None
    for i, (sub_id, t1_path, center, label01) in enumerate(plan):
        if sub_id != cur_sub:
            t1 = nib.load(t1_path).get_fdata().astype(np.float32)
            cur_sub = sub_id
            print(f"  {sub_id}...")
        z, yy, x = center
        X_mm[i, 0] = t1[z-half:z+half, yy-half:yy+half, x-half:x+half]
        y_arr[i] = label01
        s_arr[i] = sub_id

    X_mm.flush()
    np.save(y_path, y_arr)
    np.save(s_path, s_arr.astype(str))

    print(f"\n{'='*50}\nИТОГ\n{'='*50}")
    print(f"  Форма X: {X_mm.shape}")
    print(f"  Конкордантных (0): {int(np.sum(y_arr == 0))}")
    print(f"  Дискордантных (1): {int(np.sum(y_arr == 1))}")
    print(f"  Субъектов: {len(np.unique(s_arr))}")
    print(f"  X на диске: {os.path.getsize(X_path)/1e9:.2f} ГБ")
    print(f"\n  Файлы для заливки на Drive:")
    print(f"    {X_path}")
    print(f"    {y_path}")
    print(f"    {s_path}")

    print(f"\n  Патчей на субъекта:")
    uniq, counts = np.unique(s_arr, return_counts=True)
    for s, c in zip(uniq, counts):
        flag = "  ! мало" if c < 2 * PATCHES_PER_CLASS else ""
        print(f"    {s}: {c}{flag}")