#!/bin/bash
#SBATCH --job-name=medfocusleak
#SBATCH --partition=compute-gpu
#SBATCH --nodelist=compute-gpu-2-9
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/medfocus_%A_%a.out
#SBATCH --error=logs/medfocus_%A_%a.err

set -e

SPLITS=(Mammography MRI)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
echo "=========================================="
echo "=== Iniciando MedFocusLeak para: $SPLIT ==="
echo "=========================================="

mkdir -p logs

# 1. Activar entorno Miniconda y variables de cache
source ~/miniconda/etc/profile.d/conda.sh
conda activate pai2

export HF_HOME=~/hf_cache
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR=$(pwd)

# Directorio de trabajo aislado por modalidad (evita colisiones entre tareas del array)
WORK_DIR="${REPO_DIR}/work_${SPLIT}"
mkdir -p "${WORK_DIR}/white_images/1"
mkdir -p "${WORK_DIR}/bsa_output/1"
mkdir -p "${WORK_DIR}/clean_images/1"
mkdir -p "${WORK_DIR}/masks/1"
mkdir -p "${WORK_DIR}/output"

DATA_CSV="${REPO_DIR}/data_${SPLIT}.csv"
MODIFIED_CSV="${WORK_DIR}/modified_${SPLIT}.csv"

echo "=== [Paso 0/7] Preparando imagenes, mascaras y CSV base para ${SPLIT} ==="
python "${REPO_DIR}/prepare_split.py" \
    --split "${SPLIT}" \
    --repo_dir "${REPO_DIR}" \
    --work_dir "${WORK_DIR}" \
    --output_csv "${DATA_CSV}" \
    --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct"

echo "=== [Paso 1/7] Generando texto adversarial con Qwen/Qwen2.5-14B-Instruct ==="
python "${REPO_DIR}/MedFocusLeak/DataProcessing/Adv_Text.py" \
    --input_csv "${DATA_CSV}" \
    --output_csv "${MODIFIED_CSV}" \
    --k 5 \
    --model_name "Qwen/Qwen2.5-14B-Instruct"

echo "=== [Paso 2/7] Renderizando texto adversarial en imagenes blancas ==="
if [ -z "$(ls -A "${WORK_DIR}/white_images/1" 2>/dev/null)" ]; then
    python "${REPO_DIR}/MedFocusLeak/DataProcessing/White_Img.py" \
        --csv "${MODIFIED_CSV}" \
        --out_dir "${WORK_DIR}/white_images/1" \
        --limit 1000
else
    echo "Imagenes blancas ya existen en ${WORK_DIR}/white_images/1, saltando Paso 2."
fi

echo "=== [Paso 3/7] Multimodal Fusion (Target_Img_gen.py con CLIP) ==="
if [ -z "$(ls -A "${WORK_DIR}/bsa_output/1" 2>/dev/null)" ]; then
    python "${REPO_DIR}/MedFocusLeak/MultimodalFusion/Target_Img_gen.py" \
        --csv_path "${MODIFIED_CSV}" \
        --image_col images \
        --text_col Findings_5 \
        --output_dir "${WORK_DIR}/bsa_output/1" \
        --model_name "openai/clip-vit-base-patch32" \
        --eps 0.0627 \
        --alpha 0.00392 \
        --steps 100
else
    echo "BSA output ya existe en ${WORK_DIR}/bsa_output/1, saltando Paso 3."
fi

echo "=== [Paso 4/7] Ensemble Adversarial Attack (Modified_mattack) ==="
cd "${REPO_DIR}/MedFocusLeak/Modified_mattack"
python generate_adversarial_samples2.py \
    data.cle_data_path="${WORK_DIR}/clean_images" \
    data.tgt_data_path="${WORK_DIR}/white_images" \
    data.mask_data_path="${WORK_DIR}/masks" \
    data.output="${WORK_DIR}/output"

HASH_DIR=$(ls -d "${WORK_DIR}/output/img/"* | head -n 1)

echo "=== [Paso 5/7] Attention-Shift Perturbation (run_attack2.py) ==="
cd "${REPO_DIR}/MedFocusLeak/attentionshift"
python run_attack2.py \
    --image_dir "${HASH_DIR}" \
    --mask_dir "${WORK_DIR}/masks" \
    --result_dir "${HASH_DIR}" \
    --num_steps 300

echo "=== [Paso 6/7] Evaluando modelo victima Qwen/Qwen2.5-VL-7B-Instruct ==="
cd "${REPO_DIR}/MedFocusLeak/Modified_mattack"
python blackbox_text_generation.py \
    data.cle_data_path="${WORK_DIR}/clean_images" \
    data.tgt_data_path="${WORK_DIR}/white_images" \
    data.mask_data_path="${WORK_DIR}/masks" \
    data.output="${WORK_DIR}/output" \
    blackbox.model_name="Qwen/Qwen2.5-VL-7B-Instruct"

echo "=== [Paso 7/7] Evaluando con Juez Qwen/Qwen3-32B (MTR Score) ==="
python gpt_evaluate.py \
    data.cle_data_path="${WORK_DIR}/clean_images" \
    data.tgt_data_path="${WORK_DIR}/white_images" \
    data.mask_data_path="${WORK_DIR}/masks" \
    data.output="${WORK_DIR}/output" \
    blackbox.model_name="Qwen/Qwen2.5-VL-7B-Instruct" \
    judge.model_name="Qwen/Qwen3-32B" \
    judge.mode="mtr"

echo "=========================================="
echo "=== Finalizado $SPLIT. Resultados en: ${WORK_DIR}/output/export_bundle ==="
echo "=========================================="
