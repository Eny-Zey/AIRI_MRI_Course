import os
import numpy as np
import nibabel as nib

BASE_PATH  = "./ds004873"
LABELS_DIR = "./labels"

PATCH_SIZE = 16   # тот же масштаб, что у патчей (проверим окрестность этого размера)
HALF = PATCH_SIZE // 2

# Берём несколько субъектов для оценки (не обязательно все — статистика устойчива)
SUBJECTS = [
    "sub-p019", "sub-p023", "sub-p026", "sub-p036", "sub-p043",
    "sub-p050", "sub-p055",
]

SAMPLES_PER_SUBJ = 2000   # сколько вокселей проверить у каждого

def analyze_subject(sub_id, rng):
    label_path = os.path.join(LABELS_DIR, f"{sub_id}_label-concordance.nii.gz")
    if not os.path.exists(label_path):
        return None
    labels = nib.load(label_path).get_fdata().astype(np.uint8)
    shape = labels.shape

    # все размеченные воксели (1 или 2), не приграничные
    coords = np.argwhere(labels > 0)
    coords = coords[
        (coords[:,0] >= HALF) & (coords[:,0] <= shape[0]-HALF) &
        (coords[:,1] >= HALF) & (coords[:,1] <= shape[1]-HALF) &
        (coords[:,2] >= HALF) & (coords[:,2] <= shape[2]-HALF)
    ]
    if len(coords) == 0:
        return None

    rng.shuffle(coords)
    coords = coords[:SAMPLES_PER_SUBJ]

    purity_scores = []   # доля соседей с той же меткой, что у центра
    for (z, y, x) in coords:
        center_label = labels[z, y, x]
        cube = labels[z-HALF:z+HALF, y-HALF:y+HALF, x-HALF:x+HALF]
        # рассматриваем только размеченные воксели в кубе (метка > 0)
        labeled = cube[cube > 0]
        if len(labeled) == 0:
            continue
        same = np.sum(labeled == center_label)
        purity = same / len(labeled)
        purity_scores.append(purity)

    return np.array(purity_scores)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    all_purity = []

    print(f"Проверка однородности меток в окрестности {PATCH_SIZE}³\n")
    print(f"{'Субъект':<14} {'ср.чистота':<12} {'медиана':<10} {'доля чистых>0.8'}")
    print("-" * 50)

    for sub_id in SUBJECTS:
        p = analyze_subject(sub_id, rng)
        if p is None or len(p) == 0:
            print(f"{sub_id:<14} нет данных")
            continue
        all_purity.append(p)
        frac_clean = np.mean(p > 0.8)
        print(f"{sub_id:<14} {p.mean():<12.3f} {np.median(p):<10.3f} {frac_clean:.3f}")

    all_purity = np.concatenate(all_purity)
    print("-" * 50)
    print(f"\nОБЩАЯ СТАТИСТИКА (по {len(all_purity)} вокселям):")
    print(f"  Средняя чистота окрестности: {all_purity.mean():.3f}")
    print(f"  Медиана:                     {np.median(all_purity):.3f}")
    print(f"  Доля 'чистых' (>80% соседей той же метки): {np.mean(all_purity > 0.8):.3f}")
    print(f"  Доля 'хаотичных' (<60% соседей):           {np.mean(all_purity < 0.6):.3f}")

    print(f"\nИНТЕРПРЕТАЦИЯ:")
    mean_p = all_purity.mean()
    if mean_p > 0.8:
        print("  Метки ОДНОРОДНЫ на масштабе патча — соседи почти всегда")
        print("  той же метки. Задача в принципе решаема, проблема не в пятнистости.")
    elif mean_p > 0.65:
        print("  Метки УМЕРЕННО однородны — есть структура, но и шум присутствует.")
        print("  Сигнал размыт, но не уничтожен.")
    else:
        print("  Метки ПЯТНИСТЫЕ — в одном патче перемешаны обе метки почти 50/50.")
        print("  Модель видит почти одинаковые патчи с разными метками →")
        print("  задача предсказания центрального вокселя по патчу плохо определена.")
        print("  Это объясняет, почему модель не учится.")
