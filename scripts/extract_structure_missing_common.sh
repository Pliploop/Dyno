#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/acw749/Dyno}"
ROOT="${STRUCTURE_ROOT:-/gpfs/scratch/acw749/datasets/structure}"
PYTHON="${PYTHON:-/data/home/acw749/conda-envs/instruct_embed/bin/python}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/gpfs/scratch/acw749/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "${ROOT}/logs"

count_npy() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -type f -name "*.npy" 2>/dev/null | wc -l
  else
    echo 0
  fi
}

expected_count() {
  case "$1" in
    harmonix) echo 424 ;;
    salami) echo 344 ;;
    *) echo "Unknown dataset '$1'" >&2; return 2 ;;
  esac
}

run_extract() {
  local dataset="$1"
  local encoder="$2"
  local rate="$3"
  local sr="$4"
  local hop="$5"
  local target_n_samples="$6"
  local batch_size="$7"
  local num_workers="$8"
  local max_batch_chunks="$9"

  local expected
  expected="$(expected_count "${dataset}")"

  local out_dir="${ROOT}/features/${dataset}/${encoder}/${rate}"
  local current
  current="$(count_npy "${out_dir}")"

  echo "[structure-extract] $(date --iso-8601=seconds) ${dataset}/${encoder}/${rate}: ${current}/${expected} files"
  if (( current >= expected )); then
    echo "[structure-extract] skipping complete ${dataset}/${encoder}/${rate}"
    return 0
  fi

  mkdir -p "${out_dir}"

  "${PYTHON}" -m dyno.extract_dataset \
    +extract_features="${encoder}" \
    paths=apocrita \
    data.folder_path="${ROOT}/audio/${dataset}" \
    save_dir="${out_dir}" \
    root_path="${ROOT}/audio/${dataset}" \
    save=true \
    ++feature_rate="${rate}" \
    hop="${hop}" \
    data.target_sr="${sr}" \
    data.target_n_samples="${target_n_samples}" \
    batch_size="${batch_size}" \
    num_workers="${num_workers}" \
    max_batch_chunks="${max_batch_chunks}" \
    data.batch_size="${batch_size}" \
    data.num_workers="${num_workers}"

  current="$(count_npy "${out_dir}")"
  echo "[structure-extract] $(date --iso-8601=seconds) ${dataset}/${encoder}/${rate}: ${current}/${expected} files after extraction"
  if (( current < expected )); then
    echo "[structure-extract] incomplete after extraction: ${dataset}/${encoder}/${rate}" >&2
    return 1
  fi
}

build_manifest_if_ready() {
  local encoder="$1"
  local rate="$2"

  local h_count s_count
  h_count="$(count_npy "${ROOT}/features/harmonix/${encoder}/${rate}")"
  s_count="$(count_npy "${ROOT}/features/salami/${encoder}/${rate}")"

  if (( h_count < 424 || s_count < 344 )); then
    echo "[structure-extract] not building manifest for ${encoder}/${rate}; harmonix=${h_count}/424 salami=${s_count}/344"
    return 0
  fi

  echo "[structure-extract] building manifest for ${encoder}/${rate}"
  "${PYTHON}" -m dyno.prepare_structure_eval \
    --root "${ROOT}" \
    --encoder "${encoder}" \
    --rate "${rate}" \
    --require-features
}

run_rate() {
  local encoder="$1"
  local rate="$2"
  local sr="$3"
  local hop="$4"
  local target_n_samples="$5"
  local batch_size="$6"
  local num_workers="$7"
  local max_batch_chunks="$8"

  run_extract harmonix "${encoder}" "${rate}" "${sr}" "${hop}" "${target_n_samples}" "${batch_size}" "${num_workers}" "${max_batch_chunks}"
  run_extract salami "${encoder}" "${rate}" "${sr}" "${hop}" "${target_n_samples}" "${batch_size}" "${num_workers}" "${max_batch_chunks}"
  build_manifest_if_ready "${encoder}" "${rate}"
}

echo "[structure-extract] job_id=${SLURM_JOB_ID:-local}"
echo "[structure-extract] host=$(hostname)"
echo "[structure-extract] start=$(date --iso-8601=seconds)"

# MERT through 0.5 Hz is extracted; make sure the 0.5 Hz manifest exists.
build_manifest_if_ready mert 0.5hz

# Remaining high-overlap rates use single-process loading to avoid CPU OOM.
run_rate muq 2hz 24000 12000 240000 1 2 64

# Remaining MERT structure rates. MERT config must keep pool=true.
run_rate mert 1hz 24000 24000 240000 1 2 32
run_rate mert 2hz 24000 12000 240000 1 2 32

# Music2Latent structure rates.
run_rate music2latent 0.1hz 44100 441000 441000 1 2 32
run_rate music2latent 0.2hz 44100 220500 441000 1 2 32
run_rate music2latent 0.5hz 44100 88200 441000 1 2 32
run_rate music2latent 1hz 44100 44100 441000 1 2 32
run_rate music2latent 2hz 44100 22050 441000 1 2 32

echo "[structure-extract] end=$(date --iso-8601=seconds)"
