#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] Line $LINENO failed." >&2' ERR

# ============================================================
# ComfyUI Qwen model installer
#
# Supports:
# - direct URLs
# - Hugging Face token validation via HF_TOKEN
# - Civitai token validation via CIVITAI_TOKEN
# - parallel downloads with wait-n scheduling
# - aria2c acceleration when available, curl fallback otherwise
# - optional model groups toggled by env vars
#
# Recommended balance for RTX 6000 Blackwell 96GB:
# - Use fp8 base models as the default path.
# - Keep bf16 disabled unless you are deliberately trading speed and disk for marginal quality.
# - Keep lightning LoRAs enabled only when you want the fast path.
#
# Usage:
#   export HF_TOKEN='your_hf_token'               # optional but validated if set
#   export CIVITAI_TOKEN='your_civitai_token'     # optional but validated if set
#   export MAX_PARALLEL_DOWNLOADS=4
#   export DOWNLOAD_OPTIONAL_ACCELERATORS=1
#   export DOWNLOAD_BF16=0
#   bash install_comfyui_models.sh
# ============================================================

COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
MODELS_DIR="$COMFYUI_DIR/models"
MAX_PARALLEL_DOWNLOADS="${MAX_PARALLEL_DOWNLOADS:-4}"
DOWNLOAD_OPTIONAL_ACCELERATORS="${DOWNLOAD_OPTIONAL_ACCELERATORS:-0}"
DOWNLOAD_BF16="${DOWNLOAD_BF16:-0}"
DOWNLOAD_QWEN_CONTROLLER_TEXT_ENCODER="${DOWNLOAD_QWEN_CONTROLLER_TEXT_ENCODER:-1}"

# Optional tokens
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd mktemp
require_cmd mv
require_cmd ls

HAVE_ARIA2C=0

check_or_offer_aria2_install() {
  if command -v aria2c >/dev/null 2>&1; then
    HAVE_ARIA2C=1
    return 0
  fi

  echo "[WARN] aria2c is not installed. Downloads will use curl fallback (slower for very large files)."

  if [[ -t 0 ]]; then
    local reply=""
    read -r -p "[PROMPT] Install aria2 now for faster downloads? [y/N] " reply
    case "${reply,,}" in
      y|yes)
        if command -v apt-get >/dev/null 2>&1; then
          echo "[INFO] Installing aria2 via apt-get..."
          apt-get update && apt-get install -y aria2
        else
          echo "[WARN] apt-get is not available on this system. Install aria2 manually."
        fi
        ;;
      *)
        echo "[INFO] Continuing without aria2."
        ;;
    esac
  else
    echo "[INFO] Non-interactive shell: skipping install prompt."
    echo "[INFO] To enable accelerator later: apt-get update && apt-get install -y aria2"
  fi

  if command -v aria2c >/dev/null 2>&1; then
    HAVE_ARIA2C=1
    echo "[OK] aria2c is available and will be used."
  fi
}

RUNNING_DOWNLOADS=0
declare -A SCHEDULED_TARGETS=()

check_or_offer_aria2_install

mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$MODELS_DIR/loras" \
  "$MODELS_DIR/checkpoints" \
  "$MODELS_DIR/controlnet" \
  "$MODELS_DIR/ipadapter" \
  "$MODELS_DIR/clip_vision" \
  "$MODELS_DIR/animatediff_models" \
  "$MODELS_DIR/upscale_models"

build_headers() {
  local url="$1"
  local -n out_headers_ref="$2"
  out_headers_ref=()

  if [[ "$url" == *"huggingface.co"* && -n "$HF_TOKEN" ]]; then
    out_headers_ref+=("Authorization: Bearer $HF_TOKEN")
  fi

  if [[ "$url" == *"civitai.com"* && -n "$CIVITAI_TOKEN" ]]; then
    out_headers_ref+=("Authorization: Bearer $CIVITAI_TOKEN")
  fi
}

validate_hf_token() {
  if [[ -z "$HF_TOKEN" ]]; then
    echo "[WARN] HF_TOKEN is not set. Public Hugging Face URLs will still download, but private assets will fail."
    return 0
  fi

  if curl -fsS -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/whoami-v2" >/dev/null; then
    echo "[OK] HF_TOKEN validated successfully."
  else
    echo "[ERROR] HF_TOKEN is set but failed validation against the Hugging Face whoami endpoint." >&2
    exit 1
  fi
}

validate_civitai_token() {
  if [[ -z "$CIVITAI_TOKEN" ]]; then
    echo "[WARN] CIVITAI_TOKEN is not set. Public Civitai URLs will still download, but private assets will fail."
    return 0
  fi

  if curl -fsS -H "Authorization: Bearer $CIVITAI_TOKEN" "https://civitai.com/api/v1/me" >/dev/null; then
    echo "[OK] CIVITAI_TOKEN validated successfully."
  else
    echo "[ERROR] CIVITAI_TOKEN is set but failed validation against the Civitai me endpoint." >&2
    exit 1
  fi
}

download_with_aria2c() {
  local url="$1"
  local dest_dir="$2"
  local final_name="$3"
  shift 3
  local headers=("$@")

  local aria2_headers=()
  local h
  for h in "${headers[@]}"; do
    aria2_headers+=("--header=$h")
  done

  aria2c \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --split=8 \
    --min-split-size=8M \
    --max-connection-per-server=8 \
    --file-allocation=none \
    "${aria2_headers[@]}" \
    --dir="$dest_dir" \
    --out="$final_name" \
    "$url"
}

download_with_curl() {
  local url="$1"
  local dest_dir="$2"
  local final_name="$3"
  shift 3
  local headers=("$@")

  local tmp_file="$dest_dir/$final_name.part"

  local curl_headers=()
  local h
  for h in "${headers[@]}"; do
    curl_headers+=("-H" "$h")
  done

  curl -fL \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 2 \
    --continue-at - \
    "${curl_headers[@]}" \
    -o "$tmp_file" \
    "$url"

  mv "$tmp_file" "$dest_dir/$final_name"
}

download_file() {
  local url="$1"
  local dest_dir="$2"
  local final_name="$3"

  if [[ -z "$url" ]]; then
    echo "[WARN] Skipping $final_name because URL is empty."
    return 0
  fi

  mkdir -p "$dest_dir"

  local headers=()
  build_headers "$url" headers

  # Re-runs should never re-download complete files.
  if [[ -s "$dest_dir/$final_name" ]]; then
    echo "[SKIP] Already present: $dest_dir/$final_name"
    return 0
  fi

  echo "[INFO] Downloading: $final_name"
  echo "       URL: $url"
  echo "       TO:  $dest_dir/$final_name"

  if [[ $HAVE_ARIA2C -eq 1 ]]; then
    download_with_aria2c "$url" "$dest_dir" "$final_name" "${headers[@]}"
  else
    download_with_curl "$url" "$dest_dir" "$final_name" "${headers[@]}"
  fi

  echo "[OK] Saved: $dest_dir/$final_name"
  ls -lh "$dest_dir/$final_name"
  echo
}

queue_download() {
  local url="$1"
  local dest_dir="$2"
  local final_name="$3"
  local target="$dest_dir/$final_name"

  if [[ -n "${SCHEDULED_TARGETS[$target]:-}" ]]; then
    echo "[SKIP] Already queued in this run: $target"
    return 0
  fi
  SCHEDULED_TARGETS[$target]=1

  while (( RUNNING_DOWNLOADS >= MAX_PARALLEL_DOWNLOADS )); do
    wait -n
    RUNNING_DOWNLOADS=$(( RUNNING_DOWNLOADS - 1 )) || true
  done

  download_file "$url" "$dest_dir" "$final_name" &
  RUNNING_DOWNLOADS=$(( RUNNING_DOWNLOADS + 1 )) || true
}

echo "[INFO] ComfyUI directory: $COMFYUI_DIR"
echo "[INFO] Models directory:  $MODELS_DIR"
echo "[INFO] Parallel limit:    $MAX_PARALLEL_DOWNLOADS"
if [[ $HAVE_ARIA2C -eq 1 ]]; then
  echo "[INFO] Download accelerator: aria2c"
else
  echo "[INFO] Download accelerator: curl fallback"
fi

validate_hf_token
validate_civitai_token

# ------------------------------------------------------------
# Manifest: required Qwen stacks
# ------------------------------------------------------------

# Qwen-Image-2512 text-to-image stack
URL_QWEN_IMAGE_2512_FP8="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"
URL_QWEN_IMAGE_2512_BF16="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors"
URL_QWEN_25_VL_TEXT_ENCODER="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
URL_QWEN_3_8B_FP8MIXED_TEXT_ENCODER="https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen_3_8b_fp8mixed.safetensors"
URL_QWEN_IMAGE_VAE="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# Qwen-Image-2512 accelerators (optional speed path)
URL_QWEN_IMAGE_2512_LIGHTNING_4STEP="https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"

# Qwen-Image-Edit-2509 multiple-angle / reference-image stack
URL_QWEN_IMAGE_EDIT_2509_FP8="https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
URL_QWEN_IMAGE_EDIT_2509_MULTI_ANGLES="https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/loras/Qwen-Edit-2509-Multiple-angles.safetensors"
URL_QWEN_IMAGE_EDIT_2509_LIGHTNING_4STEP="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"

if [[ "$DOWNLOAD_BF16" == "1" ]]; then
  queue_download "$URL_QWEN_IMAGE_2512_BF16" "$MODELS_DIR/diffusion_models" "qwen_image_2512_bf16.safetensors"
fi

queue_download "$URL_QWEN_IMAGE_2512_FP8" "$MODELS_DIR/diffusion_models" "qwen_image_2512_fp8_e4m3fn.safetensors"
queue_download "$URL_QWEN_25_VL_TEXT_ENCODER" "$MODELS_DIR/text_encoders" "qwen_2.5_vl_7b_fp8_scaled.safetensors"

if [[ "$DOWNLOAD_QWEN_CONTROLLER_TEXT_ENCODER" == "1" ]]; then
  queue_download "$URL_QWEN_3_8B_FP8MIXED_TEXT_ENCODER" "$MODELS_DIR/text_encoders" "qwen_3_8b_fp8mixed.safetensors"
fi

queue_download "$URL_QWEN_IMAGE_VAE" "$MODELS_DIR/vae" "qwen_image_vae.safetensors"

if [[ "$DOWNLOAD_OPTIONAL_ACCELERATORS" == "1" ]]; then
  queue_download "$URL_QWEN_IMAGE_2512_LIGHTNING_4STEP" "$MODELS_DIR/loras" "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
fi

queue_download "$URL_QWEN_IMAGE_EDIT_2509_FP8" "$MODELS_DIR/diffusion_models" "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
queue_download "$URL_QWEN_IMAGE_EDIT_2509_MULTI_ANGLES" "$MODELS_DIR/loras" "Qwen-Edit-2509-Multiple-angles.safetensors"

if [[ "$DOWNLOAD_OPTIONAL_ACCELERATORS" == "1" ]]; then
  queue_download "$URL_QWEN_IMAGE_EDIT_2509_LIGHTNING_4STEP" "$MODELS_DIR/loras" "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
fi

echo
echo "[INFO] Recommended runtime presets for this GPU:"
echo "       - Draft/speed: enable lightning LoRAs, use 4 steps"
echo "       - Best balance: fp8 base models, 50 steps for 2512, 20-30 steps for edit tasks"
echo "       - Maximum fidelity: use bf16 only if you can accept the extra disk and slower load time"

wait

# ------------------------------------------------------------
# Final status summary
# ------------------------------------------------------------
echo
echo "[SUMMARY] Model file status:"

_check_model() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    local size
    size=$(du -sh "$path" 2>/dev/null | cut -f1)
    echo "  [OK]     $label ($size)"
  else
    echo "  [MISS]   $label"
  fi
}

_check_model "$MODELS_DIR/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"    "qwen_image_2512_fp8_e4m3fn.safetensors"
_check_model "$MODELS_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"         "qwen_2.5_vl_7b_fp8_scaled.safetensors"
if [[ "$DOWNLOAD_QWEN_CONTROLLER_TEXT_ENCODER" == "1" ]]; then
  _check_model "$MODELS_DIR/text_encoders/qwen_3_8b_fp8mixed.safetensors"              "qwen_3_8b_fp8mixed.safetensors"
fi
_check_model "$MODELS_DIR/vae/qwen_image_vae.safetensors"                               "qwen_image_vae.safetensors"
_check_model "$MODELS_DIR/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
_check_model "$MODELS_DIR/loras/Qwen-Edit-2509-Multiple-angles.safetensors"             "Qwen-Edit-2509-Multiple-angles.safetensors"

if [[ "$DOWNLOAD_OPTIONAL_ACCELERATORS" == "1" ]]; then
  _check_model "$MODELS_DIR/loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"       "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
  _check_model "$MODELS_DIR/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
fi

if [[ "$DOWNLOAD_BF16" == "1" ]]; then
  _check_model "$MODELS_DIR/diffusion_models/qwen_image_2512_bf16.safetensors" "qwen_image_2512_bf16.safetensors"
fi

echo
echo "[DONE] Model installation script finished."
