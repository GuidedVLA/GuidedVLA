#!/usr/bin/env bash
set -Eeuo pipefail

# Formal paper-oriented LIBERO training run.
#
# 4 x compatible GPUs (A100 80GB by default), CUDA devices 0,1,2,3 by default
# per-GPU micro-batch = 4
# physical global batch = 4 * 4 = 16
# gradient accumulation = 4
# effective global batch = 16 * 4 = 64
# activation gradient checkpointing = disabled
#
# The paper uses global batch 64. On this 4-GPU node we preserve that optimizer
# batch through accumulation; this is not a claim of bitwise H200 equivalence.
# Training precision is locked to the repository's paper-aligned path:
# FP32 parameters/gradients/AdamW states with BF16 CUDA autocast.

# Set GUIDEDVLA_BASE on each machine. The remaining locations preserve the
# prepared Fudan layout but can be overridden individually when necessary.
: "${GUIDEDVLA_BASE:?Set GUIDEDVLA_BASE to the prepared experiment root}"
BASE="${GUIDEDVLA_BASE}"
PROJECT_ROOT="${GUIDEDVLA_PROJECT_ROOT:-${BASE}/repo-libero-author-ga2-20260731}"
RUNTIME="${GUIDEDVLA_RUNTIME:-${BASE}/runtime/pi0-conversion-env}"
DATA_ROOT="${GUIDEDVLA_DATA_ROOT:-${BASE}/datasets/ybwowen-libero-477f7959}"
PI0_BASE="${GUIDEDVLA_PI0_BASE:-${BASE}/models/pi0_base_pytorch_float32}"
DEPTH_ROOT="${GUIDEDVLA_DEPTH_ROOT:-${BASE}/models/da3-small-e08cab65}"
TOKENIZER_PATH="${GUIDEDVLA_TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
ASSETS_ROOT="${GUIDEDVLA_ASSETS_ROOT:-${BASE}/assets}"
REQUIRED_GPU_MODEL="${GUIDEDVLA_REQUIRED_GPU_MODEL:-A100}"
MIN_GPU_MEMORY_BYTES="${GUIDEDVLA_MIN_GPU_MEMORY_BYTES:-79000000000}"

RUN_ID="${GUIDEDVLA_RUN_ID:-a100_4gpu_libero_plus_formal_localbatch4_ga4_gc_off_30kx2_v5_20260806}"
OUTPUT_ROOT="${BASE}/outputs/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
STATUS_PATH="${LOG_ROOT}/status.txt"
PREFLIGHT_ROOT="${LOG_ROOT}/preflight"
LAUNCHER_PATH="${PROJECT_ROOT}/manifests/$(basename "${BASH_SOURCE[0]}")"

AUTHOR_COMMIT="04be059e0d6bd448be5cb45fdbafc775f7eb5e38"
# This commit removes the old world-size-based automatic GC override.
FORMAL_CODE_COMMIT="5b33f645b2fe60c73f58b4100cb614d0a3b07daf"
DEPTH_ANYTHING_COMMIT="2c21ea849ceec7b469a3e62ea0c0e270afc3281a"
SALAD_COMMIT="6aede13a3f6c25750bf7fde10209c06cb73060bb"

STAGE1_CONFIG="pi0_libero"
STAGE2_CONFIG="pi0_libero_object_depth_skill"
STAGE1_RUN="pi0_libero_formal_4gpu_localbatch4_ga4_gc_off_30k_v5_20260806"
STAGE2_RUN="pi0_libero_full_guided_formal_4gpu_localbatch4_ga4_gc_off_30k_v5_20260806"
STAGE1_DIR="${OUTPUT_ROOT}/${STAGE1_CONFIG}/${STAGE1_RUN}"
STAGE2_DIR="${OUTPUT_ROOT}/${STAGE2_CONFIG}/${STAGE2_RUN}"
STATS1="${ASSETS_ROOT}/${STAGE1_CONFIG}/ybwowen/libero/norm_stats.json"
STATS2="${ASSETS_ROOT}/${STAGE2_CONFIG}/ybwowen/libero/norm_stats.json"

NPROC=4
LOCAL_BATCH=4
PHYSICAL_GLOBAL_BATCH=$((NPROC * LOCAL_BATCH))
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_GLOBAL_BATCH=$((PHYSICAL_GLOBAL_BATCH * GRADIENT_ACCUMULATION_STEPS))
NUM_TRAIN_STEPS=30000
SAVE_INTERVAL=10000
LOG_INTERVAL=10
VAL_INTERVAL=10000
VAL_MAX_BATCHES=1

test ! -e "${OUTPUT_ROOT}"
test ! -e "${LOG_ROOT}"
test ! -e "${CACHE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${PREFLIGHT_ROOT}" \
    "${CACHE_ROOT}/hf" "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"

exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1
printf 'RUNNING\n' >"${STATUS_PATH}"

completed=0
monitor_pid=""
on_exit() {
    exit_code=$?
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ ${exit_code} -eq 0 && ${completed} -eq 1 ]]; then
        printf 'SUCCESS\n' >"${STATUS_PATH}"
    else
        printf 'FAILED exit_code=%s completed=%s\n' "${exit_code}" "${completed}" >"${STATUS_PATH}"
    fi
}
trap on_exit EXIT

test -x "${RUNTIME}/bin/python"
test -f "${DATA_ROOT}/meta/info.json"
test -f "${DATA_ROOT}/meta/stats.json"
test -f "${DATA_ROOT}/meta/tasks.parquet"
test -f "${PI0_BASE}/model.safetensors"
test -f "${DEPTH_ROOT}/model.safetensors"
test -f "${TOKENIZER_PATH}"
test -f "${STATS1}"
test -f "${STATS2}"
test -f "${LAUNCHER_PATH}"

test "$(git -C "${PROJECT_ROOT}" merge-base "${AUTHOR_COMMIT}" HEAD)" = "${AUTHOR_COMMIT}"
test "$(git -C "${PROJECT_ROOT}" merge-base "${FORMAL_CODE_COMMIT}" HEAD)" = "${FORMAL_CODE_COMMIT}"
test "$(git -C "${PROJECT_ROOT}" hash-object scripts/train_pytorch.py)" = \
    "$(git -C "${PROJECT_ROOT}" rev-parse "${FORMAL_CODE_COMMIT}:scripts/train_pytorch.py")"
test "$(git -C "${PROJECT_ROOT}/third_party/depth_anything" rev-parse HEAD)" = "${DEPTH_ANYTHING_COMMIT}"
test "$(git -C "${PROJECT_ROOT}/third_party/depth_anything/da3_streaming/loop_utils/salad" rev-parse HEAD)" = "${SALAD_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" diff --name-only --ignore-submodules=all)"

sha256sum "${DATA_ROOT}/meta/info.json" | awk '{print $1}' | grep -qx \
    "b94a2f0abe4a75b13935869f5d8aca7e746e7473c6f8f60e35fb245d74c067e6"
sha256sum "${DATA_ROOT}/meta/stats.json" | awk '{print $1}' | grep -qx \
    "7a188908edbf8b8465b6c663c66a863fbc85cc75482a012b91a5c957ac73e926"
sha256sum "${DATA_ROOT}/meta/tasks.parquet" | awk '{print $1}' | grep -qx \
    "3166b8b5a12b19b5ae39c3300ffef15c676c3f49a5e73bf4400b8baf7b316748"
sha256sum "${STATS1}" | awk '{print $1}' | grep -qx \
    "9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855"
sha256sum "${STATS2}" | awk '{print $1}' | grep -qx \
    "9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855"
sha256sum "${PI0_BASE}/model.safetensors" | awk '{print $1}' | grep -qx \
    "c275e91bd2727e2f1651528f323548b441108e50d32d9aec679e11dc3705c362"
sha256sum "${DEPTH_ROOT}/model.safetensors" | awk '{print $1}' | grep -qx \
    "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
sha256sum "${TOKENIZER_PATH}" | awk '{print $1}' | grep -qx \
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"

export CUDA_VISIBLE_DEVICES="${GUIDEDVLA_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WANDB_MODE=disabled
export COMPILE_WARMUP_STEPS=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cat >"${PREFLIGHT_ROOT}/settings.txt" <<EOF
run_id=${RUN_ID}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
world_size=${NPROC}
gpu_model_requirement=${REQUIRED_GPU_MODEL}
minimum_gpu_memory_bytes=${MIN_GPU_MEMORY_BYTES}
per_gpu_micro_batch=${LOCAL_BATCH}
physical_global_batch=${PHYSICAL_GLOBAL_BATCH}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}
activation_gradient_checkpointing=false
num_train_steps_per_stage=${NUM_TRAIN_STEPS}
save_interval=${SAVE_INTERVAL}
log_interval=${LOG_INTERVAL}
val_interval=${VAL_INTERVAL}
val_max_batches=${VAL_MAX_BATCHES}
training_precision_config=float32
cuda_autocast=bfloat16
optimizer_state_precision=float32
stage1_config=${STAGE1_CONFIG}
stage2_config=${STAGE2_CONFIG}
stage1_init=${PI0_BASE}
stage2_init=${STAGE1_DIR}/30000
formal_code_commit=${FORMAL_CODE_COMMIT}
EOF

"${RUNTIME}/bin/python" - "${REQUIRED_GPU_MODEL}" "${MIN_GPU_MEMORY_BYTES}" "${NPROC}" <<'PY'
import sys

import torch

required_gpu_model, minimum_gpu_memory_bytes, expected_gpu_count = sys.argv[1:]
minimum_gpu_memory_bytes = int(minimum_gpu_memory_bytes)
expected_gpu_count = int(expected_gpu_count)
assert torch.cuda.device_count() == expected_gpu_count, torch.cuda.device_count()
for index in range(expected_gpu_count):
    props = torch.cuda.get_device_properties(index)
    assert required_gpu_model in props.name, props.name
    assert props.total_memory >= minimum_gpu_memory_bytes, props.total_memory
    print(f"GPU[{index}]={props.name} total_memory_bytes={props.total_memory}")
print("per_gpu_batch=4 physical_global_batch=16 GA=4 effective_global_batch=64 GC=false")
print("training_precision=FP32 parameters/gradients/AdamW states + BF16 CUDA autocast")
PY

git -C "${PROJECT_ROOT}" rev-parse HEAD >"${PREFLIGHT_ROOT}/git_commit.txt"
git -C "${PROJECT_ROOT}" diff --name-only --ignore-submodules=all \
    >"${PREFLIGHT_ROOT}/tracked_worktree_diff.txt"
git -C "${PROJECT_ROOT}" diff "${AUTHOR_COMMIT}..HEAD" \
    >"${PREFLIGHT_ROOT}/author_release_to_formal.diff"
sha256sum "${LAUNCHER_PATH}" \
    "${PROJECT_ROOT}/scripts/train_pytorch.py" \
    "${PROJECT_ROOT}/src/openpi/training/config.py" \
    >"${PREFLIGHT_ROOT}/code_sha256.txt"
sha256sum "${DATA_ROOT}/meta/info.json" "${DATA_ROOT}/meta/stats.json" \
    "${DATA_ROOT}/meta/tasks.parquet" "${STATS1}" "${STATS2}" \
    >"${PREFLIGHT_ROOT}/data_sha256.txt"
nvidia-smi -L >"${PREFLIGHT_ROOT}/gpu.txt"

(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 5
    done
) >"${LOG_ROOT}/gpu_memory.log" 2>&1 &
monitor_pid=$!

cd "${PROJECT_ROOT}"

common_args=(
    --local_root_dir "${DATA_ROOT}"
    --data.assets.asset-id ybwowen/libero
    --assets-base-dir "${ASSETS_ROOT}"
    --checkpoint-base-dir "${OUTPUT_ROOT}"
    --batch-size "${PHYSICAL_GLOBAL_BATCH}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --num-workers 2
    --pytorch-training-precision float32
    --no-wandb-enabled
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --val-interval "${VAL_INTERVAL}"
    --val-max-batches "${VAL_MAX_BATCHES}"
)

echo "===== STAGE 1 START ====="
"${RUNTIME}/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    scripts/train_pytorch.py "${STAGE1_CONFIG}" \
    --exp-name "${STAGE1_RUN}" \
    --repo_id ybwowen/libero \
    "${common_args[@]}" \
    --pytorch-weight-path "${PI0_BASE}" \
    2>&1 | tee "${LOG_ROOT}/${STAGE1_RUN}.log"

printf 'STAGE1_TRAINING_COMPLETE\n' >"${LOG_ROOT}/stage1_complete.txt"
for step in 10000 20000 30000; do
    for filename in model.safetensors optimizer.pt metadata.pt; do
        test -f "${STAGE1_DIR}/${step}/${filename}"
    done
done

echo "===== STAGE 2 START ====="
"${RUNTIME}/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    scripts/train_pytorch.py "${STAGE2_CONFIG}" \
    --exp-name "${STAGE2_RUN}" \
    "${common_args[@]}" \
    --model.depth-model-name "${DEPTH_ROOT}" \
    --pytorch-weight-path "${STAGE1_DIR}/30000" \
    2>&1 | tee "${LOG_ROOT}/${STAGE2_RUN}.log"

for step in 10000 20000 30000; do
    for filename in model.safetensors optimizer.pt metadata.pt; do
        test -f "${STAGE2_DIR}/${step}/${filename}"
    done
done

sha256sum "${STAGE1_DIR}/30000/model.safetensors" \
    "${STAGE2_DIR}/30000/model.safetensors" \
    >"${LOG_ROOT}/final_checkpoint_sha256.txt"
printf 'STAGE2_TRAINING_COMPLETE\n' >"${LOG_ROOT}/stage2_complete.txt"
printf 'TRAINING_COMPLETE\n' >"${LOG_ROOT}/TRAINING_COMPLETE"
completed=1
