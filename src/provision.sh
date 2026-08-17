#!/usr/bin/env bash
# One-time server provisioning: python env + dependencies + model weights.
# Usage (via run-remote.ps1):
#   provision.sh --env        create ~/h3-env and install packages
#   provision.sh --download   download FL2VA model set to ~/models/minimax-h3
#   provision.sh --download --with-ref2va   also fetch transformer_ref/ (Ref2VA)
set -euo pipefail

ENV_DIR="$HOME/h3-env"
MODEL_DIR="$HOME/models/minimax-h3"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # driver 550.x -> CUDA 12.4 max

do_env() {
    if [ ! -x "$ENV_DIR/bin/python" ]; then
        echo "=== creating venv at $ENV_DIR"
        if ! python3 -m venv "$ENV_DIR" 2>/dev/null; then
            echo "python3 -m venv unavailable, falling back to virtualenv"
            pip3 install --user -q virtualenv
            python3 -m virtualenv "$ENV_DIR"
        fi
    fi
    echo "=== installing packages"
    "$ENV_DIR/bin/pip" install -q -U pip setuptools wheel
    "$ENV_DIR/bin/pip" install -q --index-url "$TORCH_INDEX" torch==2.6.0 torchvision==0.21.0
    # MiniMaxH3 classes are not in any stable diffusers release yet (0.39 lacks
    # them) -> install from git main, where the H3 integration lives.
    "$ENV_DIR/bin/pip" install -q "git+https://github.com/huggingface/diffusers.git"
    "$ENV_DIR/bin/pip" install -q \
        "transformers>=4.57" "accelerate>=1.0" \
        "safetensors" "sentencepiece" "protobuf" "einops" \
        "av" "imageio" "imageio-ffmpeg" "pillow" "numpy" "psutil" \
        "huggingface_hub" "hf_transfer"
    # torchao 0.15.0: newest release that still runs on torch 2.6 (pure-python
    # fallback, cpp exts skipped) and satisfies diffusers' >=0.15 requirement.
    # Provides Int8WeightOnlyConfig(version=2) -> pinnable int8, streamed offload.
    "$ENV_DIR/bin/pip" install -q "torchao==0.15.0"
    # sageattention 1.0.6: triton int8-QK attention, ~1.4x on sm_86. Optional but
    # used by the "sage" attention_backend config option.
    "$ENV_DIR/bin/pip" install -q "sageattention==1.0.6"
    echo "=== verifying torch + CUDA + torchao"
    "$ENV_DIR/bin/python" - <<'PYEOF'
import inspect, torch
assert torch.cuda.is_available(), "CUDA not available to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0))
from torchao.quantization import Int8WeightOnlyConfig
sig = inspect.signature(Int8WeightOnlyConfig.__init__ if hasattr(Int8WeightOnlyConfig, "__init__") else Int8WeightOnlyConfig)
print("Int8WeightOnlyConfig params:", list(sig.parameters))
import diffusers, transformers
print("diffusers", diffusers.__version__, "| transformers", transformers.__version__)
from diffusers import ModularPipeline, TorchAoConfig, MiniMaxH3Transformer3DModel
from diffusers.hooks import apply_group_offloading
print("H3 classes import OK")
PYEOF
    echo "=== env ready"
}

do_download() {
    local ref2va="${1:-}"
    mkdir -p "$MODEL_DIR"
    local includes=(
        "model_index.json" "modular_model_index.json"
        "scheduler/*" "audio_scheduler/*" "tokenizer/*" "processor/*"
        "vae/*" "audio_vae/*" "text_encoder/*" "transformer/*"
    )
    if [ "$ref2va" = "--with-ref2va" ]; then
        includes+=("transformer_ref/*")
    fi
    echo "=== downloading MiniMax-H3 components to $MODEL_DIR (~134 GiB, resumable)"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    "$ENV_DIR/bin/hf" download MiniMaxAI/MiniMax-H3 \
        --include "${includes[@]}" \
        --local-dir "$MODEL_DIR"
    echo "=== download complete"
    du -sh "$MODEL_DIR"
}

case "${1:-}" in
    --env) do_env ;;
    --download) do_download "${2:-}" ;;
    *) echo "usage: provision.sh --env | --download [--with-ref2va]" >&2; exit 1 ;;
esac
