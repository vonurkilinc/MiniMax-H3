# MiniMax-H3 Remote GUI

A Windows Tkinter control panel for running MiniMax-H3 video generation on a
remote Ubuntu GPU server. Jobs are staged over SSH, generated remotely, and
downloaded into the local `outputs/` directory.

## Features

- T2VA, FL2VA, and Ref2VA workflows
- Sage or native attention
- Image, video, and audio references
- Experimental transformer LoRA loading
- Video continuation from the final frame of an existing video
- Multi-loop generation that chains each generated final frame into the next clip
- Per-loop clips plus a combined final MP4

The GUI defaults to the tested 8-step turbo/Sage configuration in
`configs/fast.json`: 960x544, 124 frames, and 9 inference steps.

## Setup

Requirements on the Windows controller:

- Python with Tkinter
- PowerShell 5.1 or newer
- OpenSSH (`ssh` and `scp`)

The default target server is already configured in `run-remote.ps1`. Then provision it:

```powershell
.\run-remote.ps1 -Provision
.\run-remote.ps1 -DownloadModel
```

`-Server user@host` can be supplied directly for command-line runs.

For Ref2VA support, download the additional transformer partition:

```powershell
.\run-remote.ps1 -DownloadModel -WithRef2VA
```

The turbo LoRA weights referenced by `configs/fast.json`,
`configs/balanced.json`, and `configs/fastest.json` are not included in this
repository. Place them in `inputs/` before using those presets.

## Run

```powershell
python h3_gui.py
```

For continuation, select a video in **Continue video** and choose the number
of generated segments in **Loops**. The first segment starts from the selected
video's final frame. Each later segment starts from the previous generated
segment's final frame. With multiple loops, numbered parts and a combined MP4
are returned under `outputs/<job-name>/`.

Generated outputs, logs, local GUI session state, input media, and model/LoRA
weights are intentionally excluded from version control.

## Simple video joiner

To concatenate existing clips in a small standalone window:

```powershell
python video_joiner.py
```

The joiner needs `ffmpeg` on `PATH`. If it is not installed system-wide, the
GUI can use the Python bundle from:

```powershell
python -m pip install imageio-ffmpeg
```

The inputs should use compatible video/audio codecs for the fast stream-copy
join. The clips are joined in the order shown in the list.
