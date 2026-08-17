# MiniMax-H3 model requirements

The repository contains the code and configuration only. Checkpoints and LoRA
weights are intentionally not committed: the base model is roughly 134 GiB,
and the local LoRAs are hundreds of MiB to over 1 GiB each. GitHub Git is not a
model-weight store.

## Base checkpoint

The remote server must have the `MiniMaxAI/MiniMax-H3` repository downloaded to:

```text
~/models/minimax-h3
```

The normal T2VA/FL2VA workflows require these components:

```text
model_index.json
modular_model_index.json
scheduler/
audio_scheduler/
tokenizer/
processor/
vae/
audio_vae/
text_encoder/
transformer/
```

Ref2VA additionally requires:

```text
transformer_ref/
```

Download the normal model set through the project runner:

```powershell
.\run-remote.ps1 -Provision
.\run-remote.ps1 -DownloadModel
```

Add the Ref2VA partition when needed:

```powershell
.\run-remote.ps1 -DownloadModel -WithRef2VA
```

## LoRA weights

Put these files in the local `inputs/` directory. The runner copies the files
into each remote job automatically.

| Filename | Used by | Notes |
| --- | --- | --- |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | `fast.json`, `balanced.json`, GUI default | Tested 8-step turbo LoRA; strength 1.0 |
| `minimax_h3_turbo_4step_ema_ckpt500_diffusers.safetensors` | `fastest.json` | 4-step turbo LoRA; strength 1.0 |
| `MiniMaxSpicy.safetensors` | Custom/local GUI session | AI Toolkit fused-layout H3 LoRA; converted automatically by the loader |

The first two are preset dependencies. `MiniMaxSpicy.safetensors` is present in
the working setup as a custom LoRA but is not referenced by a checked-in
preset. Its native `qkv_proj` tensors are split into Diffusers Q/K/V modules at
load time. Do not rename the files unless you also update the corresponding
JSON configuration.

## Python dependencies

Install the exact dependency set from [`requirements.txt`](requirements.txt),
or use the fully automated remote setup:

```powershell
.\run-remote.ps1 -Provision
```

The Sage backend requires `sageattention==1.0.6` in the tested environment.
Use `attention_backend: "native"` in a config if Sage is unavailable.
