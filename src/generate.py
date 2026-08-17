#!/usr/bin/env python3
"""MiniMax-H3 batch inference worker.

Runs on the Ubuntu GPU server inside /dev/shm/h3-job. All inputs arrive via a
JSON config and the job directory layout:

    job/
      in/config.json        parameters + prompt / prompt_file + input media names
      in/<media files>      keyframes / reference media named by the config
      out/                  results: <output_name>.mp4, manifest.json, and
                            numbered parts when loops > 1

Model weights are read from $H3_MODEL_DIR (default ~/models/minimax-h3) and are
never copied into the job dir. Nothing outside the job dir is written.

Loading strategy for a single RTX 3090 (24 GB), straight from the official
diffusers consumer-card recipe: transformer and Qwen3-VL text encoder are
quantized to int8 (torchao) while loading from the BF16 checkpoint, then their
blocks are streamed from host RAM to the GPU (group offloading). VAEs stay
resident on the GPU. Expect ~75 GB of host RAM usage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("h3")

MODEL_DIR_DEFAULT = str(Path.home() / "models" / "minimax-h3")

# modules_to_not_convert lists from the official diffusers MiniMax-H3 docs.
TRANSFORMER_NO_QUANT = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
ENCODER_NO_QUANT = [
    "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head",
]


def int8_config():
    """Int8WeightOnlyConfig(version=2) where torchao supports it (pinnable
    tensors, needed for streamed offload); fall back to plain int8."""
    from torchao.quantization import Int8WeightOnlyConfig

    try:
        return Int8WeightOnlyConfig(version=2)
    except TypeError:
        log.warning("torchao has no Int8WeightOnlyConfig(version=2); using v1 int8")
        return Int8WeightOnlyConfig()


def peak_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def apply_loras(transformer, lora_specs, in_dir: Path):
    """Attach LoRA deltas to the transformer as bf16 side modules.

    No official H3 LoRAs exist yet; this accepts the two common safetensors
    conventions (diffusers `lora_down/lora_up`, PEFT `lora_A/lora_B`, optional
    per-layer `alpha`), matched to modules by fully-qualified name. The delta
    is computed in bf16 on top of the int8-quantized base weights and is
    registered as a child module, so block-level CPU offload carries it along.
    """
    import re

    import safetensors.torch as st
    import torch

    class _LoraDelta(torch.nn.Module):
        def __init__(self, a, b, scale):
            super().__init__()
            self.a = torch.nn.Parameter(a.to(torch.bfloat16), requires_grad=False)
            self.b = torch.nn.Parameter(b.to(torch.bfloat16), requires_grad=False)
            self.scale = scale

        def forward(self, x):
            return torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.a), self.b
            ) * self.scale

    key_re = re.compile(
        r"^(?:transformer\.)?(.+?)\.(lora_down|lora_up|lora_A|lora_B)(?:\.[^.]+)?\.weight$"
    )
    alpha_re = re.compile(r"^(?:transformer\.)?(.+?)\.alpha(?:\.[^.]+)?$")
    name_to_mod = dict(transformer.named_modules())

    for i, spec in enumerate(lora_specs):
        path = in_dir / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file not found in job inputs: {spec['path']}")
        strength = float(spec.get("strength", 1.0))
        pairs, alphas = {}, {}
        with st.safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                m = key_re.match(k)
                if m:
                    fqn, kind = m.group(1), m.group(2)
                    side = "A" if kind in ("lora_down", "lora_A") else "B"
                    pairs.setdefault(fqn, {})[side] = f.get_tensor(k)
                else:
                    m2 = alpha_re.match(k)
                    if m2:
                        alphas[m2.group(1)] = float(f.get_tensor(k))
        # global alpha from file metadata (e.g. lightx2v turbo: alpha=8) is the
        # fallback when no per-layer alpha keys exist
        global_alpha = float(meta["alpha"]) if "alpha" in meta else None
        applied = skipped = 0
        for fqn, ab in sorted(pairs.items()):
            mod = name_to_mod.get(fqn)
            if mod is None or "A" not in ab or "B" not in ab:
                skipped += 1
                continue
            a, b = ab["A"], ab["B"]
            alpha = alphas.get(fqn, global_alpha)
            scale = strength * (alpha / a.shape[0] if alpha else 1.0)
            delta = _LoraDelta(a, b, scale)
            mod.add_module(f"h3_lora_{i}", delta)
            mod.register_forward_hook(
                lambda module, inputs, output, d=delta: output + d(inputs[0].to(torch.bfloat16)).to(output.dtype)
            )
            applied += 1
        log.info("LoRA %s: applied to %d modules, skipped %d unmatched (strength=%.2f)",
                 spec["path"], applied, skipped, strength)
        if applied == 0:
            raise ValueError(
                f"LoRA {spec['path']}: no keys matched transformer modules. "
                "Expected diffusers (lora_down/lora_up) or PEFT (lora_A/lora_B) key layout."
            )


def load_pipeline(model_dir: str, workflow: str, loras=None, in_dir: Path | None = None,
                  attention_backend: str | None = None):
    import torch
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Scheduler,
        MiniMaxH3Transformer3DModel,
        ModularPipeline,
        TorchAoConfig,
    )
    from diffusers.hooks import apply_group_offloading
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    # t2va and fl2va share the transformer/ partition; ref2va uses transformer_ref/.
    t_subfolder = "transformer" if workflow in ("t2va", "fl2va") else "transformer_ref"
    t_component = "transformer" if workflow in ("t2va", "fl2va") else "transformer_ref"

    if not (Path(model_dir) / t_subfolder).is_dir():
        raise FileNotFoundError(
            f"{model_dir}/{t_subfolder} is missing. "
            f"Run the model download (run-remote.ps1 -DownloadModel"
            f"{' -WithRef2VA' if t_subfolder == 'transformer_ref' else ''}) first."
        )

    log.info("creating pipeline (workflow=%s, partition=%s)", workflow, t_subfolder)
    pipe = ModularPipeline.from_pretrained(model_dir)

    log.info("loading transformer (int8) from %s/", t_subfolder)
    t0 = time.time()
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        model_dir,
        subfolder=t_subfolder,
        dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            int8_config(), modules_to_not_convert=TRANSFORMER_NO_QUANT
        ),
    )
    log.info("transformer loaded in %.0fs", time.time() - t0)

    log.info("loading text encoder (int8) from text_encoder/")
    t0 = time.time()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            int8_config(), modules_to_not_convert=ENCODER_NO_QUANT
        ),
    )
    log.info("text encoder loaded in %.0fs", time.time() - t0)

    log.info("loading remaining components (vae, audio_vae, schedulers, tokenizer, processor)")
    # NOTE: load_components() resolves component specs through the hub repo id
    # baked into modular_model_index.json, so every component is loaded
    # explicitly from the local dir instead (offline-safe).
    pipe.update_components(
        **{
            t_component: transformer,
            "text_encoder": text_encoder,
            "vae": AutoencoderKLMiniMaxH3.from_pretrained(
                model_dir, subfolder="vae", dtype=torch.bfloat16
            ),
            "audio_vae": AutoencoderKLMiniMaxH3Audio.from_pretrained(
                model_dir, subfolder="audio_vae", dtype=torch.bfloat16
            ),
            "scheduler": MiniMaxH3Scheduler.from_pretrained(model_dir, subfolder="scheduler"),
            "audio_scheduler": MiniMaxH3Scheduler.from_pretrained(
                model_dir, subfolder="audio_scheduler"
            ),
            "tokenizer": Qwen2TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer"),
            "processor": Qwen3VLProcessor.from_pretrained(model_dir, subfolder="processor"),
        }
    )

    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if loras:
        # before group offloading, so the LoRA delta params ride the block groups
        apply_loras(transformer, loras, in_dir)

    if attention_backend and attention_backend != "native":
        log.info("attention backend: %s", attention_backend)
        if "sage" in attention_backend:
            # diffusers gates sage backends behind sageattention>=2.1.1 and binds
            # `sageattn = None` at import when the check fails. The installed
            # 1.0.6 triton implementation is call-compatible for inference
            # (q/k/v, tensor_layout, is_causal, sm_scale; return_lse stays
            # False), so rebind the symbol and lift the gate.
            import diffusers.models.attention_dispatch as _ad
            if getattr(_ad, "sageattn", None) is None:
                from sageattention import sageattn as _sageattn
                _ad.sageattn = _sageattn
            if hasattr(_ad, "_CAN_USE_SAGE_ATTN"):
                _ad._CAN_USE_SAGE_ATTN = True
        transformer.set_attention_backend(attention_backend)

    # use_stream=False: streamed offload pins tensors at onload time, and
    # torchao's Int8Tensor does not implement is_pinned/pin_memory on this
    # torch 2.6 + torchao 0.15 (python-fallback) stack. Non-streamed group
    # offload copies synchronously - a bit slower, but it runs.
    # low_cpu_mem_usage=True keeps params as plain CPU tensors (no pinning).
    log.info("enabling group offload (transformer blocks, encoder leaves, stream=False)")
    offload = dict(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        use_stream=False,
        low_cpu_mem_usage=True,
    )
    transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")
    return pipe


def build_references(ref_specs, in_dir: Path):
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    classes = {
        "image": MiniMaxH3ImageReference,
        "video": MiniMaxH3VideoReference,
        "audio": MiniMaxH3AudioReference,
    }
    refs = []
    for spec in ref_specs:
        kind = spec["type"]
        if kind not in classes:
            raise ValueError(f"unknown reference type: {kind!r} (use image|video|audio)")
        path = in_dir / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"reference file not found in job inputs: {spec['path']}")
        refs.append(classes[kind].from_file(str(path)))
    return refs


def load_last_video_frame(path: Path):
    """Decode only one frame at a time and return the final frame as RGB PIL."""
    import av

    if not path.is_file():
        raise FileNotFoundError(f"continuation video not found in job inputs: {path.name}")
    last_frame = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"continuation input has no video stream: {path.name}")
        for frame in container.decode(container.streams.video[0]):
            last_frame = frame
    if last_frame is None:
        raise ValueError(f"continuation input has no decodable frames: {path.name}")
    return last_frame.to_image().convert("RGB")


def concat_video_parts(parts: list[Path], output_path: Path) -> None:
    """Losslessly concatenate identically encoded generated clips with ffmpeg."""
    import imageio_ffmpeg

    concat_list = output_path.parent / ".h3-concat.txt"
    try:
        concat_list.write_text(
            "".join(f"file '{part.resolve().as_posix()}'\n" for part in parts),
            encoding="utf-8",
        )
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    in_dir = job_dir / "in"
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workflow = cfg.get("workflow", "t2va")
    loops = int(cfg.get("loops", 1))
    if not 1 <= loops <= 100:
        raise ValueError("config 'loops' must be between 1 and 100")
    if workflow == "ref2va" and (cfg.get("continue_video") or loops > 1):
        raise ValueError("video continuation loops require the t2va/fl2va transformer")
    model_dir = os.environ.get("H3_MODEL_DIR", cfg.get("model_dir", MODEL_DIR_DEFAULT))

    prompt = cfg.get("prompt")
    if not prompt and cfg.get("prompt_file"):
        prompt = (in_dir / cfg["prompt_file"]).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("config must set 'prompt' or 'prompt_file'")

    log.info("workflow=%s frames=%s size=%sx%s steps=%s seed=%s loops=%s",
             workflow, cfg.get("num_frames"), cfg.get("height"), cfg.get("width"),
             cfg.get("num_inference_steps"), cfg.get("seed"), loops)

    manifest = {"config": cfg, "model_dir": model_dir, "ok": False}
    t_start = time.time()
    try:
        import torch
        from diffusers.utils import load_image
        from diffusers.utils.export_utils import encode_video

        t0 = time.time()
        pipe = load_pipeline(
            model_dir, workflow, cfg.get("loras"), in_dir, cfg.get("attention_backend")
        )
        manifest["load_seconds"] = round(time.time() - t0, 1)

        call = {"prompt": prompt}
        if cfg.get("image"):
            call["image"] = load_image(str(in_dir / cfg["image"]))
        if cfg.get("last_image"):
            call["last_image"] = load_image(str(in_dir / cfg["last_image"]))
        if cfg.get("references"):
            call["references"] = build_references(cfg["references"], in_dir)
        for key in ("num_frames", "height", "width", "num_inference_steps"):
            if cfg.get(key) is not None:
                call[key] = cfg[key]

        seed = cfg.get("seed")
        call["generator"] = (
            torch.Generator().manual_seed(int(seed)) if seed is not None and int(seed) >= 0
            else torch.Generator()
        )
        call["output"] = ["videos", "audio", "sampling_rate"]
        call["output_type"] = "pil"

        if cfg.get("continue_video"):
            if cfg.get("image"):
                raise ValueError("choose either config 'image' or 'continue_video', not both")
            call["image"] = load_last_video_frame(in_dir / cfg["continue_video"])
            log.info("loop 1 will continue from the last frame of %s", cfg["continue_video"])

        name = cfg.get("output_name", "h3_output")
        if name in (".", "..") or any(char in name for char in "/\\\r\n'"):
            raise ValueError("config 'output_name' must be a plain file name without slashes or quotes")
        fps = float(cfg.get("fps", 24))
        out_path = out_dir / f"{name}.mp4"
        parts = []
        loop_stats = []
        generate_seconds = 0.0
        manifest["loops_completed"] = 0
        manifest["loop_stats"] = loop_stats
        manifest["parts"] = []
        for loop_index in range(loops):
            log.info("generating loop %d/%d...", loop_index + 1, loops)
            t0 = time.time()
            results = pipe(**call)
            elapsed = time.time() - t0
            generate_seconds += elapsed

            part_path = (
                out_path if loops == 1
                else out_dir / f"{name}_part{loop_index + 1:03d}.mp4"
            )
            encode_video(
                results["videos"][0],
                fps=fps,
                output_path=str(part_path),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
            parts.append(part_path)
            loop_stats.append({
                "loop": loop_index + 1,
                "generate_seconds": round(elapsed, 1),
                "output": part_path.name,
            })
            manifest["loops_completed"] = loop_index + 1
            manifest["parts"] = [part.name for part in parts]
            manifest["generate_seconds"] = round(generate_seconds, 1)
            log.info("wrote loop %d/%d to %s", loop_index + 1, loops, part_path)

            # The default PIL output makes this directly acceptable as the
            # next FL2VA first-frame keyframe. A user-supplied last keyframe
            # only applies to loop 1; later clips continue freely.
            call["image"] = results["videos"][0][-1].copy()
            call.pop("last_image", None)

        if loops > 1:
            concat_video_parts(parts, out_path)
            log.info("combined %d loops into %s", loops, out_path)

        manifest["generate_seconds"] = round(generate_seconds, 1)
        manifest["parts"] = [part.name for part in parts]
        manifest["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        manifest["output"] = out_path.name
        manifest["ok"] = True
        log.info("wrote final output %s", out_path)
    except Exception:
        manifest["error"] = traceback.format_exc()
        log.error("generation failed:\n%s", manifest["error"])
    finally:
        manifest["total_seconds"] = round(time.time() - t_start, 1)
        manifest["peak_rss_mb"] = round(peak_rss_mb(), 1)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
