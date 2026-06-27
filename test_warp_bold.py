"""
ТЕСТ переноса BOLD T2->MNI на ОДНОМ субъекте.
Проверяем визуально (MRIcron), что BOLD-карта ложится ровно на MNI.

Цепочка: BOLD(T2) --[T2->T1w affine]--> T1w --[T1w->MNI h5]--> MNI

Запуск:
  /home/eny-zey/miniconda3/envs/MRI/bin/python test_warp_bold.py
"""
import os
import numpy as np
import ants

SUB = "sub-p019"
BASE = f"ds004873/derivatives/{SUB}"

bold_t2 = f"{BASE}/func/{SUB}_1stlevel_calccontrol_space-T2.nii.gz"
aff_t2_t1 = f"{BASE}/anat/{SUB}_desc-fmriprep_T2_to_T1w.mat"
h5_t1_mni = f"{BASE}/anat/{SUB}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5"
mni_ref = f"{BASE}/qmri/{SUB}_task-calc_space-MNI152_desc-orig_cmro2.nii.gz"

os.makedirs("reliability/test", exist_ok=True)

# читаем ASCII-матрицу 4x4 (T2->T1w)
M = np.loadtxt(aff_t2_t1)
print("Матрица T2->T1w:\n", M)

# создаём ANTs affine-трансформацию из матрицы
# ANTs хранит affine как matrix(3x3)+offset, в LPS-координатах.
# fmriprep .mat обычно в RAS; конвертация RAS->LPS: инверсия знаков X,Y.
tfm = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
# матрица в файле: world(T1w) = M @ world(T2). Берём 3x3 и сдвиг.
mat3 = M[:3, :3]
off = M[:3, 3]
# RAS->LPS флип
flip = np.diag([-1, -1, 1])
mat_lps = flip @ mat3 @ flip
off_lps = flip @ off
ants.set_ants_transform_parameters(tfm, np.concatenate([mat_lps.flatten(), off_lps]))
tfm_file = "reliability/test/t2_to_t1w.mat"
ants.write_transform(tfm, tfm_file)

bold = ants.image_read(bold_t2)
ref = ants.image_read(mni_ref)

# применяем цепочку: сначала affine T2->T1w, потом warp T1w->MNI
warped = ants.apply_transforms(
    fixed=ref, moving=bold,
    transformlist=[h5_t1_mni, tfm_file],   # порядок: последний применяется первым
    interpolator="linear",
)
out = f"reliability/test/{SUB}_BOLD_in_MNI.nii.gz"
ants.image_write(warped, out)
print(f"\nСохранено: {out}")
print(f"Шаблон для наложения: {mni_ref}")
print("\nПроверь в MRIcron:")
print(f"  подложка: {mni_ref}")
print(f"  оверлей:  {out}")
print("BOLD-активация должна лежать в коре, ровно по мозгу шаблона.")
print(f"\nДиапазон значений BOLD: [{warped.numpy().min():.2f}, {warped.numpy().max():.2f}]")
