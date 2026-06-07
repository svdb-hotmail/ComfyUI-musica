#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] Line $LINENO failed." >&2' ERR

# Install the public models required by the Yvann longform/audio-reactive workflows.
# Optional private/gated-download credentials are read only from environment variables:
#   CIVITAI_API_TOKEN
#   HUGGINGFACE_TOKEN

COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
MODELS_DIR="${MODELS_DIR:-$COMFYUI_DIR/models}"

CIVITAI_API_TOKEN="${CIVITAI_API_TOKEN:-}"
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"

downloaded_count=0
skipped_count=0
failed_count=0

required_dirs=(
  checkpoints
  animatediff_models
  loras
  upscale_models
  controlnet
  vae
  ipadapter
  clip_vision
)

for dir in "${required_dirs[@]}"; do
  mkdir -p "$MODELS_DIR/$dir"
done

build_headers() {
  local url="$1"
  local -n out_headers_ref="$2"
  out_headers_ref=()

  if [[ "$url" == *"civitai.com"* && -n "$CIVITAI_API_TOKEN" ]]; then
    out_headers_ref+=(-H "Authorization: Bearer $CIVITAI_API_TOKEN")
  fi

  if [[ "$url" == *"huggingface.co"* && -n "$HUGGINGFACE_TOKEN" ]]; then
    out_headers_ref+=(-H "Authorization: Bearer $HUGGINGFACE_TOKEN")
  fi
}

download_file() {
  local url="$1"
  local dest_subdir="$2"
  local final_name="$3"
  local dest_dir="$MODELS_DIR/$dest_subdir"
  local dest_file="$dest_dir/$final_name"

  if [[ -z "$url" ]]; then
    echo "[WARN] No URL configured for $dest_subdir/$final_name; skipping."
    return 0
  fi

  mkdir -p "$dest_dir"

  if [[ -s "$dest_file" ]]; then
    echo "[SKIP] Exists: $dest_file"
    skipped_count=$((skipped_count + 1))
    return 0
  fi

  local tmp_file="$dest_file.tmp.$$"
  local headers=()
  build_headers "$url" headers

  echo "[INFO] Downloading $dest_subdir/$final_name"
  echo "[INFO] Source: $url"

  if curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 "${headers[@]}" -o "$tmp_file" "$url"; then
    mv "$tmp_file" "$dest_file"
    chmod 0644 "$dest_file"
    downloaded_count=$((downloaded_count + 1))
    echo "[OK] Saved: $dest_file"
    ls -lh "$dest_file"
  else
    rm -f "$tmp_file"
    failed_count=$((failed_count + 1))
    echo "[FAIL] Could not download: $dest_subdir/$final_name" >&2
    return 1
  fi

  echo
}

echo "[INFO] ComfyUI dir: $COMFYUI_DIR"
echo "[INFO] Models dir:  $MODELS_DIR"
echo "[INFO] Tokens: HUGGINGFACE_TOKEN=$([[ -n "$HUGGINGFACE_TOKEN" ]] && echo set || echo unset), CIVITAI_API_TOKEN=$([[ -n "$CIVITAI_API_TOKEN" ]] && echo set || echo unset)"
echo

download_file "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors" \
  checkpoints \
  DreamShaper_8_pruned.safetensors

download_file "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt" \
  animatediff_models \
  AnimateLCM_sd15_t2v.ckpt

download_file "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors" \
  loras \
  AnimateLCM_sd15_t2v_lora.safetensors

download_file "https://huggingface.co/Cobacabo/c00l/resolve/main/sudo_UltraCompact_2x_1.121.175_G.pth" \
  upscale_models \
  2x-sudo-UltraCompact.pth

download_file "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors" \
  controlnet \
  control_v1p_sd15_qrcode_monster_v2.safetensors

download_file "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors" \
  controlnet \
  control_v11f1p_sd15_depth_fp16.safetensors

download_file "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors" \
  controlnet \
  control_v11p_sd15_lineart_fp16.safetensors

download_file "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt" \
  controlnet \
  v3_sd15_sparsectrl_rgb.ckpt

download_file "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
  vae \
  vae-ft-mse-840000-ema-pruned.safetensors

download_file "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt" \
  loras \
  v3_sd15_adapter.ckpt

download_file "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" \
  ipadapter \
  ip-adapter-plus_sd15.safetensors

download_file "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
  clip_vision \
  CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors

echo "[INFO] Workflow references LiquidAF-0-1.safetensors, but no public source URL is configured in this repo."
echo "[INFO] If you have that file, place it under: $MODELS_DIR/animatediff_models or $MODELS_DIR/loras, depending on the AnimateDiff node expectation."
echo
echo "[DONE] Model installation finished. Downloaded: $downloaded_count, skipped existing: $skipped_count, failed: $failed_count"

if [[ "$failed_count" -ne 0 ]]; then
  exit 1
fi