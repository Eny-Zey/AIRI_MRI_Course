import nibabel as nib
import numpy as np

SUB = "sub-p019"
BASE = f"ds004873/derivatives/{SUB}"

files = {
    "BOLD (space-T2)": f"{BASE}/func/{SUB}_1stlevel_calccontrol_space-T2.nii.gz",
    "T2 structural":   f"{BASE}/anat/{SUB}_space-T2_T2map.nii.gz",
    "T1w (fmriprep)":  f"{BASE}/anat/{SUB}_desc-fmriprep_T1w.nii.gz",
    "CMRO2 MNI":       f"{BASE}/qmri/{SUB}_task-calc_space-MNI152_desc-orig_cmro2.nii.gz",
}

for name, path in files.items():
    try:
        img = nib.load(path)
        print(f"\n{name}")
        print(f"  shape: {img.shape}")
        print(f"  zooms: {tuple(round(float(z),2) for z in img.header.get_zooms()[:3])}")
        aff = img.affine
        print(f"  affine diag: {[round(float(aff[i,i]),2) for i in range(3)]}")
        print(f"  origin: {[round(float(aff[i,3]),1) for i in range(3)]}")
        # ориентация
        print(f"  orientation: {''.join(nib.aff2axcodes(aff))}")
    except Exception as e:
        print(f"\n{name}: НЕТ ФАЙЛА ({e})")
