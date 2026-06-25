#!/bin/bash

DS="ds004873"
ROOT="./${DS}"
S3="https://s3.amazonaws.com/openneuro.org/${DS}"

SUBJECTS=(
    sub-p019 sub-p020 sub-p021 sub-p023 sub-p026 sub-p027 sub-p028
    sub-p030 sub-p031 sub-p032 sub-p033 sub-p034 sub-p035 sub-p036
    sub-p037 sub-p038 sub-p039 sub-p040 sub-p043 sub-p044 sub-p046
    sub-p047 sub-p048 sub-p049 sub-p050 sub-p051 sub-p052 sub-p054
    sub-p055 sub-p058 sub-p059 sub-p060 sub-p061 sub-p063 sub-p064
    sub-p065 sub-p066 sub-p067 sub-p068
)

download_if_missing() {
    local url=$1
    local dest=$2
    if [ -f "$dest" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    wget -q --show-progress -O "$dest" "$url"
    if [ $? -ne 0 ]; then
        echo "  ✗ Ошибка: $dest"
        rm -f "$dest"
        return 1
    fi
}

is_complete() {
    local s=$1
    [ -f "${ROOT}/derivatives/${s}/anat/${s}_desc-fmriprep_T1w.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/anat/${s}_desc-fmriprep_brain_mask.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/func/${s}_1stlevel_calccontrol_space-T2.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/qmri/${s}_task-calc_space-T1w_desc-orig_cmro2.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/qmri/${s}_task-control_space-T1w_desc-orig_cmro2.nii.gz" ]
}

INCOMPLETE=()

for SUB in "${SUBJECTS[@]}"; do
    echo "=================================================="
    if is_complete "$SUB"; then
        echo "✓ $SUB — уже полный, пропускаю"
        continue
    fi
    echo "Скачиваю $SUB ..."

    ANAT="${ROOT}/derivatives/${SUB}/anat"
    FUNC="${ROOT}/derivatives/${SUB}/func"
    QMRI="${ROOT}/derivatives/${SUB}/qmri"

    download_if_missing \
        "${S3}/derivatives/${SUB}/anat/${SUB}_desc-fmriprep_T1w.nii.gz" \
        "${ANAT}/${SUB}_desc-fmriprep_T1w.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/anat/${SUB}_desc-fmriprep_brain_mask.nii.gz" \
        "${ANAT}/${SUB}_desc-fmriprep_brain_mask.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/func/${SUB}_1stlevel_calccontrol_space-T2.nii.gz" \
        "${FUNC}/${SUB}_1stlevel_calccontrol_space-T2.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/qmri/${SUB}_task-calc_space-T1w_desc-orig_cmro2.nii.gz" \
        "${QMRI}/${SUB}_task-calc_space-T1w_desc-orig_cmro2.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/qmri/${SUB}_task-control_space-T1w_desc-orig_cmro2.nii.gz" \
        "${QMRI}/${SUB}_task-control_space-T1w_desc-orig_cmro2.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/qmri/${SUB}_task-calc_space-T1w_cbf.nii.gz" \
        "${QMRI}/${SUB}_task-calc_space-T1w_cbf.nii.gz"

    download_if_missing \
        "${S3}/derivatives/${SUB}/qmri/${SUB}_task-control_space-T1w_cbf.nii.gz" \
        "${QMRI}/${SUB}_task-control_space-T1w_cbf.nii.gz"

    if is_complete "$SUB"; then
        echo "✓ $SUB — готово"
    else
        echo "✗ $SUB — не хватает файлов"
        INCOMPLETE+=("$SUB")
    fi
done

echo "=================================================="
echo "ИТОГ:"
if [ ${#INCOMPLETE[@]} -eq 0 ]; then
    echo "Все субъекты скачаны ✓"
else
    echo "Неполные:"
    for s in "${INCOMPLETE[@]}"; do echo "  - $s"; done
fi
