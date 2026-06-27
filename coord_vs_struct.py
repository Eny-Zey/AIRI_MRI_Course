"""
Проверка: несёт ли структура T1w информацию СВЕРХ локализации?

Сравнивает в LOSO три набора признаков на одних и тех же вокселях:
  1. только координаты (нормированная позиция в мозге)
  2. только структурные признаки патча (интенсивность, текстура)
  3. координаты + структура

Если (3) заметно лучше (1) и (2) сам по себе > 0.5 — структура несёт
информацию сверх расположения. Иначе сигнал в основном от локализации.

Структурные признаки извлекаются после z-нормировки T1w ПО СУБЪЕКТУ
(аналог поканальной нормировки в CNN, но с сохранением сопоставимости).

Запуск:
  /home/eny-zey/miniconda3/envs/MRI/bin/python coordinate_vs_structure.py
"""
import os
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
    return np.sum(lab == labels[z, y, x]) / len(lab) if len(lab) else 0.0

def patch_features(vol, z, y, x, half):
    """Структурные признаки патча (vol уже z-нормирован по субъекту)."""
    p = vol[z-half:z+half, y-half:y+half, x-half:x+half]
    gz, gy, gx = np.gradient(p)
    grad = np.sqrt(gz**2 + gy**2 + gx**2)
    return [
        p.mean(), p.std(), p.min(), p.max(),
        np.median(p), np.percentile(p, 25), np.percentile(p, 75),
        grad.mean(), grad.std(),   # текстура (резкость границ)
    ]

def collect(rng):
    coords, struct, labels_out, subj_out = [], [], [], []
    half = PATCH_SIZE // 2
    hp = max(half, PURITY_WIN // 2)

    for sub_id in ALL_SUBJECTS:
        label_path = os.path.join(LABELS_DIR, f"{sub_id}_label-concordance.nii.gz")
        t1_path = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                    f"{sub_id}_desc-fmriprep_T1w.nii.gz")
        if not (os.path.exists(label_path) and os.path.exists(t1_path)):
            print(f"  пропуск {sub_id}"); continue

        labels = nib.load(label_path).get_fdata().astype(np.uint8)
        vol = nib.load(t1_path).get_fdata().astype(np.float32)
        shape = labels.shape

        # z-нормировка T1w по субъекту (внутри мозга)
        bm_path = os.path.join(BASE_PATH, "derivatives", sub_id, "anat",
                    f"{sub_id}_desc-fmriprep_brain_mask.nii.gz")
        if os.path.exists(bm_path):
            bm = nib.load(bm_path).get_fdata() > 0
        else:
            bm = labels > 0
        mu, sd = vol[bm].mean(), vol[bm].std()
        vol = (vol - mu) / (sd + 1e-6)

        bb = np.argwhere(bm)
        mins = bb.min(0).astype(np.float32); maxs = bb.max(0).astype(np.float32)
        extent = np.maximum(maxs - mins, 1)

        for cls in (1, 2):
            cc = np.argwhere(labels == cls)
            if len(cc) == 0: continue
            valid = cc[(cc[:,0]>=hp)&(cc[:,0]<=shape[0]-hp)&
                       (cc[:,1]>=hp)&(cc[:,1]<=shape[1]-hp)&
                       (cc[:,2]>=hp)&(cc[:,2]<=shape[2]-hp)]
            if len(valid)==0: continue
            rng.shuffle(valid)
            lab01 = 0 if cls==1 else 1
            got = 0
            for c in valid:
                if got >= PATCHES_PER_CLASS: break
                z,y,x = int(c[0]),int(c[1]),int(c[2])
                if neighborhood_purity(labels,z,y,x,PURITY_WIN) < PURITY_THRESH: continue
                coords.append((np.array([z,y,x],dtype=np.float32)-mins)/extent)
                struct.append(patch_features(vol,z,y,x,half))
                labels_out.append(lab01); subj_out.append(sub_id); got += 1
        print(f"  {sub_id}: {sum(1 for s in subj_out if s==sub_id)}")

    return (np.array(coords,dtype=np.float32), np.array(struct,dtype=np.float32),
            np.array(labels_out), np.array(subj_out))

def loso(X, y, subj):
    aucs = []
    for ts in np.unique(subj):
        te = (subj==ts); tr = ~te
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], prob) if len(np.unique(y[te]))>1 else np.nan)
    a = np.array(aucs); a = a[~np.isnan(a)]
    return a

if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    print("Сбор координат + структурных признаков (те же воксели)...")
    C, S, y, subj = collect(rng)
    print(f"\nВокселей: {len(y)}, признаков структуры: {S.shape[1]}")

    print("\n  Прогон LOSO (RandomForest)...")
    a_coord  = loso(C, y, subj)
    a_struct = loso(S, y, subj)
    a_both   = loso(np.hstack([C, S]), y, subj)

    print(f"\n{'='*55}")
    print(f"  Только КООРДИНАТЫ:        AUC {a_coord.mean():.3f} ± {a_coord.std(ddof=1):.3f}")
    print(f"  Только СТРУКТУРА:         AUC {a_struct.mean():.3f} ± {a_struct.std(ddof=1):.3f}")
    print(f"  КООРДИНАТЫ + СТРУКТУРА:   AUC {a_both.mean():.3f} ± {a_both.std(ddof=1):.3f}")
    print(f"{'='*55}")
    print(f"\n  Прирост от добавления структуры к координатам: "
          f"{a_both.mean()-a_coord.mean():+.3f}")
    if a_both.mean() - a_coord.mean() > 0.02 and a_struct.mean() > 0.55:
        print("  ВЫВОД: структура несёт информацию СВЕРХ локализации.")
    elif a_struct.mean() < 0.53:
        print("  ВЫВОД: структура сама по себе почти не предсказывает —")
        print("         сигнал в основном от локализации.")
    else:
        print("  ВЫВОД: структура добавляет мало сверх координат.")
