"""Simple, responsive Tkinter desktop interface."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, BooleanVar, StringVar, Tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .artifacts import format_timestamp
from .cli import MODEL_CHOICES
from .errors import AudioSplitterError, CancelledError
from .models import PipelineOptions, PipelineResult
from .pipeline import default_output_directory, split_audio


class AudioSplitterApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("音频按句子拆分工具")
        self.root.geometry("980x720")
        self.root.minsize(820, 600)

        self.input_var = StringVar()
        self.output_var = StringVar()
        self.model_var = StringVar(value="small")
        self.language_var = StringVar(value="auto")
        self.device_var = StringVar(value="auto")
        self.format_var = StringVar(value="wav")
        self.gap_var = StringVar(value="0.8")
        self.max_duration_var = StringVar(value="20.0")
        self.padding_before_var = StringVar(value="0.12")
        self.padding_after_var = StringVar(value="0.18")
        self.prompt_var = StringVar()
        self.offline_var = BooleanVar(value=False)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.closing = False
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=BOTH, expand=True)

        files = ttk.LabelFrame(outer, text="文件", padding=8)
        files.pack(fill=X)
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="输入音频：").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(files, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(files, text="选择…", command=self._choose_input).grid(row=0, column=2)
        ttk.Label(files, text="输出目录：").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(files, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(files, text="选择…", command=self._choose_output).grid(row=1, column=2)

        settings = ttk.LabelFrame(outer, text="识别与拆分设置", padding=8)
        settings.pack(fill=X, pady=(8, 0))
        for column in range(8):
            settings.columnconfigure(column, weight=1 if column % 2 else 0)
        ttk.Label(settings, text="模型：").grid(row=0, column=0, sticky="e")
        ttk.Combobox(
            settings, textvariable=self.model_var, values=MODEL_CHOICES, state="readonly", width=15
        ).grid(row=0, column=1, sticky="ew", padx=(4, 12))
        ttk.Label(settings, text="语言：").grid(row=0, column=2, sticky="e")
        ttk.Combobox(
            settings,
            textvariable=self.language_var,
            values=("auto", "zh", "en", "ja", "ko", "fr", "de", "es"),
            width=9,
        ).grid(row=0, column=3, sticky="ew", padx=(4, 12))
        ttk.Label(settings, text="设备：").grid(row=0, column=4, sticky="e")
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=9,
        ).grid(row=0, column=5, sticky="ew", padx=(4, 12))
        ttk.Label(settings, text="格式：").grid(row=0, column=6, sticky="e")
        ttk.Combobox(
            settings,
            textvariable=self.format_var,
            values=("wav", "flac", "mp3"),
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="ew", padx=(4, 0))

        ttk.Label(settings, text="静音分句(秒)：").grid(row=1, column=0, sticky="e", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.gap_var, width=8).grid(
            row=1, column=1, sticky="ew", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(settings, text="最长句(秒)：").grid(row=1, column=2, sticky="e", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.max_duration_var, width=8).grid(
            row=1, column=3, sticky="ew", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(settings, text="句前留白：").grid(row=1, column=4, sticky="e", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.padding_before_var, width=8).grid(
            row=1, column=5, sticky="ew", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(settings, text="句后留白：").grid(row=1, column=6, sticky="e", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.padding_after_var, width=8).grid(
            row=1, column=7, sticky="ew", padx=(4, 0), pady=(7, 0)
        )

        ttk.Label(settings, text="提示词：").grid(row=2, column=0, sticky="e", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.prompt_var).grid(
            row=2, column=1, columnspan=5, sticky="ew", padx=(4, 12), pady=(7, 0)
        )
        ttk.Checkbutton(settings, text="仅离线模型", variable=self.offline_var).grid(
            row=2, column=6, columnspan=2, sticky="w", pady=(7, 0)
        )

        actions = ttk.Frame(outer)
        actions.pack(fill=X, pady=8)
        self.start_button = ttk.Button(actions, text="开始识别并拆分", command=self._start)
        self.start_button.pack(side=LEFT)
        self.cancel_button = ttk.Button(
            actions, text="取消", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side=LEFT, padx=6)
        self.open_button = ttk.Button(
            actions, text="打开输出目录", command=self._open_output, state="disabled"
        )
        self.open_button.pack(side=LEFT)
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(side=RIGHT, fill=X, expand=True, padx=(18, 0))

        result_frame = ttk.LabelFrame(outer, text="逐句结果", padding=5)
        result_frame.pack(fill=BOTH, expand=True)
        columns = ("index", "start", "end", "text", "file")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=10)
        headings = {
            "index": "编号",
            "start": "开始",
            "end": "结束",
            "text": "识别文字",
            "file": "输出文件",
        }
        widths = {"index": 55, "start": 90, "end": 90, "text": 430, "file": 150}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], stretch=column in {"text", "file"})
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        details = ttk.Panedwindow(outer, orient="horizontal")
        details.pack(fill=BOTH, expand=True, pady=(8, 0))
        transcript_frame = ttk.LabelFrame(details, text="完整识别文字", padding=5)
        log_frame = ttk.LabelFrame(details, text="运行日志", padding=5)
        self.transcript_text = ScrolledText(transcript_frame, height=7, wrap="word")
        self.log_text = ScrolledText(log_frame, height=7, wrap="word")
        self.transcript_text.pack(fill=BOTH, expand=True)
        self.log_text.pack(fill=BOTH, expand=True)
        details.add(transcript_frame, weight=2)
        details.add(log_frame, weight=1)

    def _choose_input(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择包含人声的音频或视频",
            filetypes=(
                ("音频和视频", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma *.mp4 *.mkv *.mov"),
                ("所有文件", "*.*"),
            ),
        )
        if filename:
            path = Path(filename)
            self.input_var.set(str(path))
            self.output_var.set(str(default_output_directory(path)))

    def _choose_output(self) -> None:
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_var.set(directory)

    def _make_options(self, overwrite: bool) -> PipelineOptions:
        input_path = Path(self.input_var.get().strip())
        output_value = self.output_var.get().strip()
        language = self.language_var.get().strip().lower()
        return PipelineOptions(
            input_path=input_path,
            output_dir=Path(output_value) if output_value else None,
            model=self.model_var.get(),
            language=None if not language or language == "auto" else language,
            device=self.device_var.get(),
            output_format=self.format_var.get(),
            sentence_gap=float(self.gap_var.get()),
            max_sentence_duration=float(self.max_duration_var.get()),
            padding_before=float(self.padding_before_var.get()),
            padding_after=float(self.padding_after_var.get()),
            initial_prompt=self.prompt_var.get().strip() or None,
            offline=self.offline_var.get(),
            overwrite=overwrite,
        )

    @staticmethod
    def _has_existing_outputs(output_dir: Path) -> bool:
        if not output_dir.is_dir():
            return False
        if (output_dir / "segments.json").exists() or (output_dir / "transcript.txt").exists():
            return True
        return any(output_dir.glob("[0-9][0-9][0-9]_*"))

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            preliminary = self._make_options(overwrite=False)
            if not preliminary.input_path.is_file():
                raise ValueError("请选择有效的输入文件")
            output_dir = preliminary.output_dir or default_output_directory(preliminary.input_path)
            overwrite = False
            if self._has_existing_outputs(output_dir):
                overwrite = messagebox.askyesno(
                    "确认覆盖",
                    "输出目录中已有本工具生成的文件。是否覆盖所有同名结果？",
                    parent=self.root,
                )
                if not overwrite:
                    return
            options = self._make_options(overwrite=overwrite)
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.transcript_text.delete("1.0", END)
        self.log_text.delete("1.0", END)
        self.cancel_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.progress.start(12)

        def work() -> None:
            try:
                result = split_audio(
                    options,
                    status_callback=lambda message: self.events.put(("status", message)),
                    cancel_event=self.cancel_event,
                )
                self.events.put(("result", result))
            except CancelledError as exc:
                self.events.put(("cancelled", exc))
            except AudioSplitterError as exc:
                self.events.put(("error", exc))
            except Exception as exc:  # Keep unexpected worker failures visible to the user.
                self.events.put(("error", RuntimeError(f"未预期错误：{exc}")))

        self.worker = threading.Thread(target=work, name="audio-splitter-worker", daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self._append_log("正在取消，请等待当前模型或 FFmpeg 操作结束……")

    def _append_log(self, message: str) -> None:
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)

    def _show_result(self, result: PipelineResult) -> None:
        for index, sentence in enumerate(result.sentences, start=1):
            self.tree.insert(
                "",
                END,
                values=(
                    index,
                    format_timestamp(sentence.start),
                    format_timestamp(sentence.end),
                    sentence.text,
                    sentence.output_path.name if sentence.output_path else "",
                ),
            )
        self.transcript_text.insert(END, "\n".join(sentence.text for sentence in result.sentences))
        self.output_var.set(str(result.output_dir))
        self.open_button.configure(state="normal")
        for warning in result.warnings:
            self._append_log("警告：" + warning)
        messagebox.showinfo(
            "处理完成",
            f"共生成 {len(result.sentences)} 句。\n输出目录：{result.output_dir}",
            parent=self.root,
        )

    def _finish_job(self) -> None:
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self._append_log(str(payload))
                elif kind == "result":
                    self._finish_job()
                    self._show_result(payload)  # type: ignore[arg-type]
                elif kind == "cancelled":
                    self._finish_job()
                    self._append_log("任务已取消。")
                elif kind == "error":
                    self._finish_job()
                    self._append_log("错误：" + str(payload))
                    messagebox.showerror("处理失败", str(payload), parent=self.root)
        except queue.Empty:
            pass

        if self.closing:
            if self.worker is None or not self.worker.is_alive():
                self.root.destroy()
                return
        self.root.after(100, self._poll_events)

    def _open_output(self) -> None:
        path = Path(self.output_var.get())
        if not path.is_dir():
            messagebox.showerror("目录不存在", str(path), parent=self.root)
            return
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("任务正在运行", "取消当前任务并退出？", parent=self.root):
                return
            self.cancel_event.set()
            self.closing = True
            for child in self.root.winfo_children():
                try:
                    child.configure(state="disabled")
                except Exception:
                    pass
            return
        self.root.destroy()


def main() -> None:
    root = Tk()
    try:
        ttk.Style(root).theme_use("vista" if sys.platform == "win32" else "clam")
    except Exception:
        pass
    AudioSplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
