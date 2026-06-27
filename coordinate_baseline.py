"""
Контрольный эксперимент: предсказание метки по ОДНИМ КООРДИНАТАМ вокселя
(без структуры). Проверяет, насколько сигнал объясняется локализацией.

Берёт ТЕ ЖЕ воксели, что и extract_patches.py (тот же seed, фильтр чистоты),
нормирует координаты относительно границ мозга каждого субъекта,
обучает RandomForest в LOSO-схеме и сравнивает AUC с патчевыми моделями.

Запуск локально:
  /home/eny-zey/miniconda3/envs/MRI/bin/python coordinate_baseline.py
"""
import os
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# --- те же параметры, что в extract_patches.py для 16³ чистых ---
BASE_PATH   = "./ds004873"
LABELS_DIR  = "./labels"
PATCH_SIZE        = 16
PATCHES_PER_CLASS = 2500
RANDOM_SEED       = 42
PURITY_THRESH = 0.8
PURITY_WIN    = 16

ALL_SUBJECTS = [
    "sub-p019","sub-p020","sub-p021","sub-p023","sub-p026","sub-p027","sub-p028",
    "sub-p030","sub-p031","sub-p032","sub-p033","sub-p034","sub-p035","sub-p036",
    "sub-p037","sub-p038","sub-p039","sub-p040","sub-p043","sub-p044","sub-p046",
    "sub-p047","sub-p048","sub-p049","sub-p050","sub-p051","sub-p052","sub-p054","sub-p055",
]

def neighborhood_purity(labels, z, y, x, win):
    h = win // 2
    cube = labels[z-h:z+h, y-h:y+h, x-h:x+h]
    lab = cube[cube > 0]
    if len(lab) == 0:
        return 0.0
    return np.sum(lab == labels[z, y, x]) / len(lab)

def collect(rng):
    """Собирает (subj, нормированные xyz, метка) для тех же вокселей."""
    coords_list, labels_list, subj_list = [], [], []
    half = PATCH_SIZE // 2
    hp = max(half, PURITY_WIN // 2)

    for sub_id in ALL_SUBJECTS:
        label_path = os.path.join(LABELS_DIR, f"{sub_id}_label-concordance.nii.gz")
        if not os.path.exists(label_path):
            print(f"  пропуск {sub_id}: нет меток")
            continue
        labels = nib.load(label_path).get_fdata().astype(np.uint8)
        shape = labels.shape

        # границы мозга для нормировки: bbox всех размеченных вокселей
        brain_mask_path = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                            f"{sub_id}_desc-fmriprep_brain_mask.nii.gz")
        if os.path.exists(brain_mask_path):
            bm = nib.load(brain_mask_path).get_fdata() > 0
            bb = np.argwhere(bm)
        else:
            bb = np.argwhere(labels > 0)   # fallback
        mins = bb.min(axis=0).astype(np.float32)
        maxs = bb.max(axis=0).astype(np.float32)
        extent = np.maximum(maxs - mins, 1)

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
                if neighborhood_purity(labels, z, y, x, PURITY_WIN) < PURITY_THRESH:
                    continue
                # нормированные координаты (доля пути через мозг)
                norm = (np.array([z, y, x], dtype=np.float32) - mins) / extent
                coords_list.append(norm)
                labels_list.append(label01)
                subj_list.append(sub_id)
                collected += 1
        print(f"  {sub_id}: собрано {sum(1 for s in subj_list if s == sub_id)}")

    return (np.array(coords_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int64),
            np.array(subj_list))

def loso(X, y, subj, model_name):
    subjects = np.unique(subj)
    aucs = []
    print(f"\n  LOSO с моделью: {model_name}")
    for ts in subjects:
        te = (subj == ts); tr = ~te
        if model_name == "rf":
            clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                         n_jobs=-1, random_state=0)
        else:
            clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], prob))
        else:
            aucs.append(np.nan)
    return np.array(aucs), subjects

if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    print("Сбор координат тех же вокселей, что в патчах 16³ чистых...")
    X, y, subj = collect(rng)
    print(f"\nВсего вокселей: {len(y)}, конкорд {int((y==0).sum())}, дискорд {int((y==1).sum())}")

    for mname in ("logreg", "rf"):
        aucs, subjects = loso(X, y, subj, mname)
        valid = aucs[~np.isnan(aucs)]
        print(f"\n  === Координатный baseline ({mname}) ===")
        print(f"  Средний AUC: {valid.mean():.3f} ± {valid.std(ddof=1):.3f}")
        print(f"  min={valid.min():.3f}  max={valid.max():.3f}")
        print(f"  Фолдов выше 0.55: {int((valid>0.55).sum())}/{len(valid)}")

    print("\n  Для сравнения, патчевые модели (структура):")
    print("    Simple3DCNN 16³ чистые:        AUC 0.583")
    print("    Simple3DCNN 32³ чистые +TTA:   AUC 0.596")
    print("    ResNet3D    16³ чистые +TTA:   AUC 0.594")
