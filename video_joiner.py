#!/usr/bin/env python3
"""Small Tkinter GUI for concatenating videos with FFmpeg.

The join uses FFmpeg's concat demuxer and stream-copies the inputs, so it is
fast and does not reduce quality when the clips have compatible streams.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


VIDEO_TYPES = [
    ("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"),
    ("All files", "*.*"),
]


def find_ffmpeg() -> str | None:
    """Return an FFmpeg executable from PATH or imageio-ffmpeg if installed."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def concat_file_line(path: Path) -> str:
    """Format a path for an FFmpeg concat-demuxer list file."""
    # The concat format uses single-quoted POSIX paths even on Windows.
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def join_videos(videos: list[Path], output: Path, ffmpeg: str) -> None:
    """Join videos with stream copy. Raises CalledProcessError on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", prefix="video-join-", delete=False
    ) as handle:
        concat_file = Path(handle.name)
        handle.writelines(concat_file_line(path) for path in videos)

    try:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output),
        ]
        subprocess.run(command, check=True)
    finally:
        concat_file.unlink(missing_ok=True)


class VideoJoiner(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video joiner")
        self.geometry("720x430")
        self.minsize(600, 330)
        self.videos: list[Path] = []
        self.worker: threading.Thread | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Add clips in the order they should play, then choose Join.",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(outer)
        list_frame.pack(fill="both", expand=True)
        self.video_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.video_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_list.yview)
        scrollbar.pack(side="left", fill="y")
        self.video_list.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(list_frame)
        controls.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(controls, text="Add videos...", command=self.add_videos).pack(fill="x", pady=2)
        ttk.Button(controls, text="Remove selected", command=self.remove_selected).pack(fill="x", pady=2)
        ttk.Button(controls, text="Move up", command=lambda: self.move_selected(-1)).pack(fill="x", pady=2)
        ttk.Button(controls, text="Move down", command=lambda: self.move_selected(1)).pack(fill="x", pady=2)
        ttk.Button(controls, text="Clear", command=self.clear).pack(fill="x", pady=2)

        output = ttk.Frame(outer)
        output.pack(fill="x", pady=(10, 0))
        ttk.Label(output, text="Output:").pack(side="left")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "joined.mp4"))
        ttk.Entry(output, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(output, text="Browse...", command=self.choose_output).pack(side="left")

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.join_button = ttk.Button(bottom, text="Join videos", command=self.start_join)
        self.join_button.pack(side="right")

    def add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select videos",
            filetypes=VIDEO_TYPES,
        )
        for raw_path in paths:
            path = Path(raw_path)
            if path not in self.videos:
                self.videos.append(path)
                self.video_list.insert(tk.END, str(path))
        self.status_var.set(f"{len(self.videos)} video(s) selected")

    def remove_selected(self) -> None:
        selected = list(self.video_list.curselection())
        for index in reversed(selected):
            self.video_list.delete(index)
            del self.videos[index]
        self.status_var.set(f"{len(self.videos)} video(s) selected")

    def move_selected(self, direction: int) -> None:
        selected = list(self.video_list.curselection())
        if len(selected) != 1:
            return
        old_index = selected[0]
        new_index = old_index + direction
        if not 0 <= new_index < len(self.videos):
            return
        moved = self.videos[old_index]
        self.videos[old_index], self.videos[new_index] = self.videos[new_index], self.videos[old_index]
        self.video_list.delete(old_index)
        self.video_list.insert(new_index, str(moved))
        self.video_list.selection_set(new_index)
        self.video_list.activate(new_index)

    def clear(self) -> None:
        self.videos.clear()
        self.video_list.delete(0, tk.END)
        self.status_var.set("Ready")

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save joined video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            initialfile="joined.mp4",
        )
        if path:
            self.output_var.set(path)

    def start_join(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if len(self.videos) < 2:
            messagebox.showwarning("Nothing to join", "Select at least two videos.")
            return
        output = Path(self.output_var.get().strip()).expanduser()
        if not output.name:
            messagebox.showwarning("Output missing", "Choose an output file.")
            return
        if output.resolve() in {path.resolve() for path in self.videos}:
            messagebox.showwarning("Invalid output", "The output must be different from the input videos.")
            return
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            messagebox.showerror(
                "FFmpeg not found",
                "Install FFmpeg and add it to PATH, or install imageio-ffmpeg with:\n\n"
                "python -m pip install imageio-ffmpeg",
            )
            return

        videos = list(self.videos)
        self.join_button.configure(state="disabled")
        self.status_var.set("Joining videos...")
        self.worker = threading.Thread(
            target=self._join_worker, args=(videos, output, ffmpeg), daemon=True
        )
        self.worker.start()

    def _join_worker(self, videos: list[Path], output: Path, ffmpeg: str) -> None:
        try:
            join_videos(videos, output, ffmpeg)
        except subprocess.CalledProcessError:
            self.after(
                0,
                lambda: self._join_finished(
                    False,
                    "FFmpeg could not stream-copy these clips. They may use incompatible codecs or dimensions.",
                ),
            )
        except Exception as error:  # keep errors visible without crashing the GUI thread
            self.after(0, lambda: self._join_finished(False, str(error)))
        else:
            self.after(0, lambda: self._join_finished(True, f"Saved {output}"))

    def _join_finished(self, success: bool, message: str) -> None:
        self.join_button.configure(state="normal")
        self.status_var.set(message)
        if success:
            messagebox.showinfo("Done", message)
        else:
            messagebox.showerror("Join failed", message)


if __name__ == "__main__":
    app = VideoJoiner()
    app.mainloop()
