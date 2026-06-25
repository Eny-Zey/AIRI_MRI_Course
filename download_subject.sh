#!/bin/bash
# Скачивание и очистка данных ds004873 для разметки concordant/discordant

DS="ds004873"
ROOT="./${DS}"

SUBJECTS=(
    sub-p019 sub-p020 sub-p021 sub-p023 sub-p026 sub-p027 sub-p028
    sub-p029 sub-p030 sub-p031 sub-p032 sub-p033 sub-p034 sub-p035
    sub-p036 sub-p037 sub-p038 sub-p039 sub-p040 sub-p043 sub-p044
    sub-p046 sub-p047 sub-p048 sub-p049 sub-p050 sub-p051 sub-p052
    sub-p054 sub-p055 sub-p058 sub-p059 sub-p060 sub-p061 sub-p063
    sub-p064 sub-p065 sub-p066 sub-p067 sub-p068
)

# Какие файлы в qmri оставляем (всё прочее удаляется)
keep_qmri_file() {
    case "$1" in
        *task-calc_space-T1w_desc-orig_cmro2.nii.gz)    return 0 ;;
        *task-control_space-T1w_desc-orig_cmro2.nii.gz) return 0 ;;
        *task-calc_space-T1w_cbf.nii.gz)                return 0 ;;
        *task-control_space-T1w_cbf.nii.gz)             return 0 ;;
        *) return 1 ;;
    esac
}

# Проверка: все ли обязательные файлы на месте
is_complete() {
    local s=$1
    [ -f "${ROOT}/${s}/anat/${s}_T1w.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/anat/${s}_desc-fmriprep_brain_mask.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/func/${s}_1stlevel_calccontrol_space-T2.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/qmri/${s}_task-calc_space-T1w_desc-orig_cmro2.nii.gz" ] && \
    [ -f "${ROOT}/derivatives/${s}/qmri/${s}_task-control_space-T1w_desc-orig_cmro2.nii.gz" ]
}

INCOMPLETE=()

for SUB in "${SUBJECTS[@]}"; do
    echo "=================================================="
    if is_complete "$SUB"; then
        echo "✓ $SUB уже скачан полностью — пропускаю"
        continue
    fi
    echo "Скачиваю $SUB ..."

    # 1) Точечные файлы (надёжно — единичные пути)
    openneuro-py download --dataset=$DS \
        --include="${SUB}/anat/${SUB}_T1w.nii.gz" \
        --include="derivatives/${SUB}/anat/${SUB}_desc-fmriprep_brain_mask.nii.gz" \
        --include="derivatives/${SUB}/func/${SUB}_1stlevel_calccontrol_space-T2.nii.gz"

    # 2) Папка qmri целиком — гарантированно получаем оба cmro2 + оба cbf
    openneuro-py download --dataset=$DS \
        --include="derivatives/${SUB}/qmri/"

    # 3) Чистим qmri — оставляем только нужное
    QDIR="${ROOT}/derivatives/${SUB}/qmri"
    if [ -d "$QDIR" ]; then
        for f in "$QDIR"/*; do
            [ -f "$f" ] || continue
            if ! keep_qmri_file "$f"; then
                rm -f "$f"
            fi
        done
    fi

    # 4) Чистим func от тяжёлых полных BOLD-рядов (если просочились)
    FDIR="${ROOT}/derivatives/${SUB}/func"
    if [ -d "$FDIR" ]; then
        find "$FDIR" -type f -name "*task-all*bold.nii.gz" -delete
    fi

    # 5) Проверка
    if is_complete "$SUB"; then
        echo "✓ $SUB — все нужные файлы на месте"
    else
        echo "✗ $SUB — ЧЕГО-ТО НЕ ХВАТАЕТ"
        INCOMPLETE+=("$SUB")
    fi
    echo "Размер derivatives/${SUB}: $(du -sh ${ROOT}/derivatives/${SUB} 2>/dev/null | cut -f1)"
done

echo "=================================================="
echo "ГОТОВО."
if [ ${#INCOMPLETE[@]} -eq 0 ]; then
    echo "Все субъекты скачаны полностью ✓"
else
    echo "Неполные субъекты (нужно разобраться вручную):"
    for s in "${INCOMPLETE[@]}"; do echo "  - $s"; done
fi
