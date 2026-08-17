#!/usr/bin/env python3
"""MiniMax-H3 remote control panel.

Local Windows GUI (stdlib tkinter only) that drives run-remote.ps1: builds a
job config from the widgets, ships inputs/keyframes/references/LoRAs to the
server, streams the remote log back, and collects results under outputs\.

Run:  python h3_gui.py        (from C:\\AI\\MiniMax-H3)

Notes:
- One job at a time (the server job dir /dev/shm/h3-job is a single slot).
- LoRA support is generic (diffusers/PEFT safetensors key layouts); no
  official H3 LoRAs exist yet - untested against real H3 LoRA files.
- ref2va needs the transformer_ref partition: run-remote.ps1 -DownloadModel -WithRef2VA
- Continuation loops reuse one loaded pipeline and chain each generated last
  frame into the next clip's FL2VA first-frame input.
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
RUN_PS1 = ROOT / "run-remote.ps1"
CONFIGS = ROOT / "configs"
PROMPTS = ROOT / "prompts"
OUTPUTS = ROOT / "outputs"

# Exact settings from the successful outputs/ab-turbo8-sage3 run.
GUI_DEFAULT_CONFIG = CONFIGS / "fast.json"

FPS = 24  # H3 native frame rate

# (label, width, height); all multiples of 32, short edge 544 (fast) or 768 (trained)
RESOLUTION_PRESETS = [
    ("16:9  fast  960x544", 960, 544),
    ("16:9  hd    1344x768", 1344, 768),
    ("9:16  fast  544x960", 544, 960),
    ("9:16  hd    768x1344", 768, 1344),
    ("1:1   fast  544x544", 544, 544),
    ("1:1   hd    768x768", 768, 768),
    ("4:3   fast  736x544", 736, 544),
    ("4:3   hd    1024x768", 1024, 768),
    ("3:4   fast  544x736", 544, 736),
    ("3:4   hd    768x1024", 768, 1024),
    ("21:9  fast  1280x544", 1280, 544),
    ("21:9  hd    1792x768", 1792, 768),
    ("custom", None, None),
]

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".webm", ".mkv"}
AUD_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# measured on the RTX 3090 setup: ~41 s/step steady-state at 960x544 + overhead
EST_SECONDS_PER_STEP = 41
EST_OVERHEAD_SECONDS = 240


def snap_frames(seconds: float) -> int:
    """H3 decodable frame counts are 17*n + 5; duration must stay 5-15 s."""
    frames = int(round(seconds * FPS))
    n = max(0, math.ceil((frames - 5) / 17))
    while 17 * n + 5 > 15 * FPS:  # never exceed the 15 s generation window
        n -= 1
    return 17 * n + 5


class H3Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MiniMax-H3 remote")
        self.minsize(860, 720)
        self.queue = queue.Queue()
        self.proc = None
        self.references = []  # list of (type, abspath)
        self.loras = []       # list of (abspath, strength)

        self._build_widgets()
        self._load_defaults()
        self.after(200, self._drain_queue)

    # ---------------- UI construction ----------------
    def _build_widgets(self):
        pad = dict(padx=6, pady=3, sticky="w")

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Workflow:").pack(side="left")
        self.workflow = ttk.Combobox(top, values=["t2va", "fl2va", "ref2va"],
                                     state="readonly", width=8)
        self.workflow.set("t2va")
        self.workflow.pack(side="left", padx=4)
        self.workflow.bind("<<ComboboxSelected>>", lambda e: self._sync_workflow())
        ttk.Label(top, text="Attention:").pack(side="left", padx=(12, 0))
        self.backend = ttk.Combobox(top, values=["sage", "native"],
                                    state="readonly", width=7)
        self.backend.set("sage")
        self.backend.pack(side="left", padx=4)
        self.hint = ttk.Label(top, text="", foreground="#666")
        self.hint.pack(side="left", padx=12)

        # prompt
        pf = ttk.LabelFrame(self, text="Prompt")
        pf.pack(fill="both", padx=6, pady=4, expand=False)
        self.prompt = tk.Text(pf, height=7, wrap="word")
        self.prompt.pack(fill="both", expand=True, padx=4, pady=4)
        prow = ttk.Frame(pf)
        prow.pack(fill="x")
        ttk.Button(prow, text="Load from prompts\\...",
                   command=self._load_prompt_file).pack(side="left", padx=4, pady=2)
        ttk.Button(prow, text="Save to prompts\\...",
                   command=self._save_prompt_file).pack(side="left", padx=4)

        # video parameters
        vf = ttk.LabelFrame(self, text="Video parameters")
        vf.pack(fill="x", padx=6, pady=4)
        ttk.Label(vf, text="Duration (s):").grid(row=0, column=0, **pad)
        self.duration = tk.DoubleVar(value=5.0)
        dur_sb = ttk.Spinbox(vf, from_=5.0, to=15.0, increment=0.5, width=6,
                             textvariable=self.duration,
                             command=self._sync_frames)
        dur_sb.grid(row=0, column=1, **pad)
        dur_sb.bind("<KeyRelease>", lambda e: self._sync_frames())
        self.frames_lbl = ttk.Label(vf, text="")
        self.frames_lbl.grid(row=0, column=2, **pad)

        ttk.Label(vf, text="Resolution:").grid(row=1, column=0, **pad)
        self.preset = ttk.Combobox(vf, values=[p[0] for p in RESOLUTION_PRESETS],
                                   state="readonly", width=22)
        self.preset.set(RESOLUTION_PRESETS[0][0])
        self.preset.grid(row=1, column=1, columnspan=2, **pad)
        self.preset.bind("<<ComboboxSelected>>", lambda e: self._sync_preset())
        ttk.Label(vf, text="W x H:").grid(row=1, column=3, **pad)
        self.width = tk.IntVar(value=960)
        self.height = tk.IntVar(value=544)
        ttk.Entry(vf, textvariable=self.width, width=6).grid(row=1, column=4, **pad)
        ttk.Entry(vf, textvariable=self.height, width=6).grid(row=1, column=5, **pad)

        ttk.Label(vf, text="Steps:").grid(row=2, column=0, **pad)
        self.steps = tk.IntVar(value=30)
        ttk.Spinbox(vf, from_=4, to=100, width=6, textvariable=self.steps,
                    command=self._sync_est).grid(row=2, column=1, **pad)
        ttk.Label(vf, text="Seed (-1 random):").grid(row=2, column=2, **pad)
        self.seed = tk.IntVar(value=42)
        ttk.Entry(vf, textvariable=self.seed, width=8).grid(row=2, column=3, **pad)
        ttk.Label(vf, text="Output name:").grid(row=2, column=4, **pad)
        self.out_name = tk.StringVar(value="h3_output")
        ttk.Entry(vf, textvariable=self.out_name, width=12).grid(row=2, column=5, **pad)

        ttk.Label(vf, text="Loops:").grid(row=3, column=0, **pad)
        self.loops = tk.IntVar(value=1)
        loop_sb = ttk.Spinbox(vf, from_=1, to=20, width=6, textvariable=self.loops,
                              command=self._sync_est)
        loop_sb.grid(row=3, column=1, **pad)
        loop_sb.bind("<KeyRelease>", lambda e: self._sync_est())
        ttk.Label(vf, text="each extra loop starts from the prior clip's last frame",
                  foreground="#666").grid(row=3, column=2, columnspan=4, **pad)
        self.est_lbl = ttk.Label(vf, text="", foreground="#666")
        self.est_lbl.grid(row=4, column=0, columnspan=6, **pad)

        # conditioning inputs (fl2va) and references (ref2va)
        cf = ttk.LabelFrame(self, text="Conditioning (fl2va: keyframes / ref2va: references)")
        cf.pack(fill="x", padx=6, pady=4)
        ttk.Label(cf, text="First frame:").grid(row=0, column=0, **pad)
        self.first_image = tk.StringVar()
        ttk.Entry(cf, textvariable=self.first_image, width=52).grid(row=0, column=1, **pad)
        ttk.Button(cf, text="Browse",
                   command=lambda: self._pick_image(self.first_image)).grid(row=0, column=2, **pad)
        ttk.Label(cf, text="Last frame:").grid(row=1, column=0, **pad)
        self.last_image = tk.StringVar()
        ttk.Entry(cf, textvariable=self.last_image, width=52).grid(row=1, column=1, **pad)
        ttk.Button(cf, text="Browse",
                   command=lambda: self._pick_image(self.last_image)).grid(row=1, column=2, **pad)

        ttk.Label(cf, text="Continue video:").grid(row=2, column=0, **pad)
        self.continue_video = tk.StringVar()
        ttk.Entry(cf, textvariable=self.continue_video, width=52).grid(row=2, column=1, **pad)
        ttk.Button(cf, text="Browse",
                   command=self._pick_continue_video).grid(row=2, column=2, **pad)

        ttk.Label(cf, text="References:").grid(row=3, column=0, **{**pad, "sticky": "nw"})
        self.ref_list = tk.Listbox(cf, height=3)
        self.ref_list.grid(row=3, column=1, **{**pad, "sticky": "ew"})
        rb = ttk.Frame(cf)
        rb.grid(row=3, column=2, sticky="ns")
        ttk.Button(rb, text="Add", command=self._add_reference).pack(fill="x", pady=1)
        ttk.Button(rb, text="Remove", command=lambda: self._remove_selected(
            self.ref_list, self.references)).pack(fill="x", pady=1)

        # LoRAs
        lf = ttk.LabelFrame(self, text="LoRAs (experimental - applied to the transformer, strength-scaled)")
        lf.pack(fill="x", padx=6, pady=4)
        self.lora_list = tk.Listbox(lf, height=3)
        self.lora_list.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        lb = ttk.Frame(lf)
        lb.pack(side="left", fill="y", padx=4)
        ttk.Button(lb, text="Add", command=self._add_lora).pack(fill="x", pady=1)
        ttk.Button(lb, text="Remove", command=lambda: self._remove_selected(
            self.lora_list, self.loras)).pack(fill="x", pady=1)

        # run controls
        rf = ttk.Frame(self)
        rf.pack(fill="x", padx=6, pady=4)
        ttk.Label(rf, text="Job name:").pack(side="left")
        self.job_name = tk.StringVar()
        ttk.Entry(rf, textvariable=self.job_name, width=22).pack(side="left", padx=4)
        self.run_btn = ttk.Button(rf, text="Run on server", command=self.run_job)
        self.run_btn.pack(side="left", padx=8)
        ttk.Button(rf, text="Save config...", command=self._save_config).pack(side="left", padx=2)
        ttk.Button(rf, text="Load config...", command=self._load_config).pack(side="left", padx=2)
        ttk.Button(rf, text="Open outputs folder",
                   command=lambda: os.startfile(OUTPUTS)).pack(side="left", padx=8)
        ttk.Label(rf, text="Past jobs:").pack(side="left", padx=(16, 2))
        self.jobs = ttk.Combobox(rf, state="readonly", width=24)
        self.jobs.pack(side="left")
        ttk.Button(rf, text="Open", command=self._open_job).pack(side="left", padx=2)
        self._refresh_jobs()

        # log
        self.logw = ScrolledText(self, height=12, state="disabled",
                                 background="#111", foreground="#ddd")
        self.logw.pack(fill="both", expand=True, padx=6, pady=4)

        self._sync_workflow()
        self._sync_frames()

    # ---------------- widget helpers ----------------
    def _log(self, text):
        self.logw.configure(state="normal")
        self.logw.insert("end", text)
        self.logw.see("end")
        self.logw.configure(state="disabled")

    def _sync_workflow(self):
        wf = self.workflow.get()
        hints = {
            "t2va": "text only initially; continuation loops auto-switch to FL2VA",
            "fl2va": "uses first/last frame if given",
            "ref2va": "needs transformer_ref partition on the server",
        }
        self.hint.configure(text=hints[wf])

    def _sync_preset(self):
        label = self.preset.get()
        for name, w, h in RESOLUTION_PRESETS:
            if name == label and w:
                self.width.set(w)
                self.height.set(h)

    def _sync_frames(self):
        try:
            sec = float(self.duration.get())
        except (tk.TclError, ValueError):
            return
        sec = min(15.0, max(5.0, sec))
        frames = snap_frames(sec)
        self.frames_lbl.configure(
            text=f"= {frames} frames @ {FPS} fps (~{frames / FPS:.1f} s)")
        self._sync_est()

    def _sync_est(self):
        try:
            steps = int(self.steps.get())
            loops = int(self.loops.get())
        except (tk.TclError, ValueError):
            return
        est = max(1, loops) * steps * EST_SECONDS_PER_STEP + EST_OVERHEAD_SECONDS
        self.est_lbl.configure(
            text=f"rough estimate on the 3090: ~{est // 60} min for {max(1, loops)} loop(s) "
                 "(model loads once)")

    def _pick_image(self, var):
        p = filedialog.askopenfilename(
            initialdir=ROOT / "inputs",
            filetypes=[("images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if p:
            var.set(p)

    def _pick_continue_video(self):
        p = filedialog.askopenfilename(
            initialdir=ROOT / "inputs",
            filetypes=[("videos", "*.mp4 *.mov *.webm *.mkv")])
        if p:
            self.continue_video.set(p)

    def _add_reference(self):
        paths = filedialog.askopenfilenames(
            initialdir=ROOT / "inputs",
            filetypes=[("media", "*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.webm *.mkv *.wav *.mp3 *.flac *.ogg *.m4a")])
        for p in paths:
            ext = Path(p).suffix.lower()
            kind = ("image" if ext in IMG_EXT else
                    "video" if ext in VID_EXT else
                    "audio" if ext in AUD_EXT else None)
            if not kind:
                messagebox.showwarning("Unsupported file", f"{p}\n(unknown media extension)")
                continue
            self.references.append((kind, p))
            self.ref_list.insert("end", f"{kind}: {Path(p).name}")

    def _add_lora(self):
        p = filedialog.askopenfilename(
            initialdir=ROOT / "inputs",
            filetypes=[("LoRA safetensors", "*.safetensors")])
        if not p:
            return
        strength = simpledialog.askfloat("LoRA strength", "Strength (0-2 typical):",
                                         initialvalue=1.0, minvalue=0.0, maxvalue=4.0)
        if strength is None:
            return
        self.loras.append((p, strength))
        self.lora_list.insert("end", f"{Path(p).name}  x{strength:g}")

    def _remove_selected(self, listbox, store):
        for idx in reversed(listbox.curselection()):
            listbox.delete(idx)
            del store[idx]

    def _refresh_jobs(self):
        if OUTPUTS.is_dir():
            names = sorted((d.name for d in OUTPUTS.iterdir() if d.is_dir()), reverse=True)
            self.jobs.configure(values=names)

    def _open_job(self):
        name = self.jobs.get()
        if name:
            os.startfile(OUTPUTS / name)

    # ---------------- config ----------------
    def collect_config(self):
        sec = min(15.0, max(5.0, float(self.duration.get())))
        w, h = int(self.width.get()), int(self.height.get())
        loops = int(self.loops.get())
        if w % 32 or h % 32:
            raise ValueError("width and height must be multiples of 32")
        if not (256 <= w <= 1920 and 256 <= h <= 1920):
            raise ValueError("width/height out of sane range (256-1920)")
        if not 1 <= loops <= 100:
            raise ValueError("loops must be between 1 and 100")
        if self.continue_video.get() and self.first_image.get():
            raise ValueError("choose either a first frame or a continuation video, not both")
        if (self.continue_video.get() or loops > 1) and self.workflow.get() == "ref2va":
            raise ValueError("video continuation loops require the t2va/fl2va transformer")
        if self.continue_video.get():
            video_path = Path(self.continue_video.get())
            if video_path.suffix.lower() not in VID_EXT:
                raise ValueError("continuation video must be mp4, mov, webm, or mkv")
        output_name = self.out_name.get().strip() or "h3_output"
        if (output_name in (".", "..") or
                any(char in output_name for char in "/\\\r\n'")):
            raise ValueError("output name must be a plain file name without slashes or quotes")
        cfg = {
            "workflow": self.workflow.get(),
            "attention_backend": self.backend.get(),
            "prompt": self.prompt.get("1.0", "end").strip(),
            "num_frames": snap_frames(sec),
            "height": h,
            "width": w,
            "num_inference_steps": int(self.steps.get()),
            "seed": int(self.seed.get()),
            "fps": FPS,
            "output_name": output_name,
            "loops": loops,
        }
        if not cfg["prompt"]:
            raise ValueError("prompt is empty")
        if cfg["workflow"] == "fl2va":
            if self.first_image.get():
                cfg["image"] = Path(self.first_image.get()).name
            if self.last_image.get():
                cfg["last_image"] = Path(self.last_image.get()).name
        if cfg["workflow"] == "ref2va":
            if not self.references:
                raise ValueError("ref2va needs at least one reference file")
            cfg["references"] = [{"type": t, "path": Path(p).name}
                                 for t, p in self.references]
        if self.continue_video.get():
            cfg["continue_video"] = Path(self.continue_video.get()).name
        if self.loras:
            cfg["loras"] = [{"path": Path(p).name, "strength": s}
                            for p, s in self.loras]
        return cfg

    def _input_paths(self):
        paths = []
        if self.workflow.get() == "fl2va":
            paths += [v.get() for v in (self.first_image, self.last_image) if v.get()]
        if self.workflow.get() == "ref2va":
            paths += [p for _, p in self.references]
        if self.continue_video.get():
            paths.append(self.continue_video.get())
        paths += [p for p, _ in self.loras]
        return paths

    def _save_config(self):
        try:
            cfg = self.collect_config()
        except ValueError as e:
            messagebox.showerror("Invalid config", str(e))
            return
        p = filedialog.asksaveasfilename(initialdir=CONFIGS, defaultextension=".json",
                                         filetypes=[("json", "*.json")])
        if p:
            Path(p).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _load_config(self):
        p = filedialog.askopenfilename(initialdir=CONFIGS,
                                       filetypes=[("json", "*.json")])
        if not p:
            return
        cfg = json.loads(Path(p).read_text(encoding="utf-8"))
        self._apply_config(cfg)

    def _input_path(self, name):
        """Resolve a config input basename to inputs/ when it exists locally."""
        if not name:
            return ""
        path = Path(name)
        local = ROOT / "inputs" / path.name
        return str(local if local.is_file() else path)

    def _apply_config(self, cfg):
        self.workflow.set(cfg.get("workflow", "t2va"))
        if cfg.get("attention_backend"):
            self.backend.set(cfg["attention_backend"])
        if cfg.get("prompt"):
            self.prompt.delete("1.0", "end")
            self.prompt.insert("1.0", cfg["prompt"])
        if cfg.get("num_frames"):
            # Stay within the interval that snap_frames() maps back to this
            # aligned frame count (124 frames must not round up to 141).
            self.duration.set(round(max(5.0, (cfg["num_frames"] - 0.25) / FPS), 2))
        if cfg.get("width"):
            self.width.set(cfg["width"])
            self.preset.set("custom")
        if cfg.get("height"):
            self.height.set(cfg["height"])
        if cfg.get("num_inference_steps"):
            self.steps.set(cfg["num_inference_steps"])
        if cfg.get("seed") is not None:
            self.seed.set(cfg["seed"])
        if cfg.get("output_name"):
            self.out_name.set(cfg["output_name"])
        self.loops.set(cfg.get("loops", 1))
        self.first_image.set(self._input_path(cfg.get("image")))
        self.last_image.set(self._input_path(cfg.get("last_image")))
        self.continue_video.set(self._input_path(cfg.get("continue_video")))

        self.references.clear()
        self.ref_list.delete(0, "end")
        for spec in cfg.get("references", []):
            item = (spec["type"], self._input_path(spec["path"]))
            self.references.append(item)
            self.ref_list.insert("end", f"{item[0]}: {Path(item[1]).name}")

        self.loras.clear()
        self.lora_list.delete(0, "end")
        for spec in cfg.get("loras", []):
            item = (self._input_path(spec["path"]), float(spec.get("strength", 1.0)))
            self.loras.append(item)
            self.lora_list.insert("end", f"{Path(item[0]).name}  x{item[1]:g}")
        self._sync_workflow()
        self._sync_frames()

    def _load_prompt_file(self):
        p = filedialog.askopenfilename(initialdir=PROMPTS,
                                       filetypes=[("text", "*.txt")])
        if p:
            self.prompt.delete("1.0", "end")
            self.prompt.insert("1.0", Path(p).read_text(encoding="utf-8"))

    def _save_prompt_file(self):
        p = filedialog.asksaveasfilename(initialdir=PROMPTS, defaultextension=".txt",
                                         filetypes=[("text", "*.txt")])
        if p:
            Path(p).write_text(self.prompt.get("1.0", "end").strip(), encoding="utf-8")

    def _load_defaults(self):
        default = GUI_DEFAULT_CONFIG
        if default.is_file():
            try:
                cfg = json.loads(default.read_text(encoding="utf-8"))
                self._apply_config(cfg)
            except Exception:
                pass
        example = PROMPTS / "example.txt"
        if example.is_file():
            self.prompt.insert("1.0", example.read_text(encoding="utf-8"))

    # ---------------- job execution ----------------
    def run_job(self):
        if self.proc is not None:
            messagebox.showinfo("Job running", "A job is already running (single job slot on the server).")
            return
        try:
            cfg = self.collect_config()
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid config", str(e))
            return

        name = self.job_name.get().strip() or "gui-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg_path = CONFIGS / "_gui_last.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(RUN_PS1), "-Config", f"configs\\{cfg_path.name}",
                "-JobName", name]
        inputs = self._input_paths()
        if inputs:
            args += ["-Inputs"] + inputs  # -Inputs last: binds all remaining args

        self.logw.configure(state="normal")
        self.logw.delete("1.0", "end")
        self.logw.configure(state="disabled")
        self._log(f"== starting job '{name}' ==\n")
        self.run_btn.configure(state="disabled")

        self.proc = subprocess.Popen(
            args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace",
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.queue.put(line)
        self.queue.put(None)  # sentinel: process ended

    def _drain_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item is None:
                    rc = self.proc.wait()
                    self.proc = None
                    self.run_btn.configure(state="normal")
                    self._log(f"\n== job finished, exit code {rc} ==\n")
                    self._refresh_jobs()
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.after(200, self._drain_queue)


if __name__ == "__main__":
    app = H3Gui()
    app.mainloop()
