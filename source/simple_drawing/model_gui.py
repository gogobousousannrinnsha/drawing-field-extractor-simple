"""Small Portable model manager; Tk owns the UI and workers own operations."""
from __future__ import annotations

from datetime import datetime
import ctypes
from pathlib import Path
from queue import Empty, Queue
import threading
import traceback
import sys
from typing import Any, Callable


ACTION_LABELS = {
    "refresh": "状態更新", "install": "導入", "check": "動作確認", "select": "使用モデルの変更",
}
BUTTON_LABELS = {
    "refresh": "状態更新", "install": "導入する", "check": "動作確認", "select": "このモデルを使う",
}
STAGE_LABELS = {"download": "ダウンロード", "verify": "ファイル確認", "check": "動作確認", "complete": "完了"}


def progress_message(value: dict) -> str:
    """Translate the display only; the manager retains the original progress record."""
    stage = STAGE_LABELS.get(value.get("stage"), "処理中")
    status = str(value.get("status", "")).strip()
    lower = status.lower()
    messages = {
        "pulling manifest": "モデル情報を取得しています。",
        "verifying sha256 digest": "モデルファイルの整合性を確認しています。",
        "writing manifest": "モデル情報を保存しています。",
        "success": "完了しました。",
        "モデルファイルのSHA-256を検査しています。": "モデルファイルの整合性を確認しています。",
    }
    if lower.startswith("pulling ") and lower != "pulling manifest":
        status = "モデルファイルを取得しています。"
    else:
        status = messages.get(lower, messages.get(status, status))
    return f"{stage}：{status}" if status else stage


def format_size(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "未取得"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return "未取得"


def perform_action(manager, action: str, profile_id: str | None, cancel, progress):
    """Use the same manager operations as the CLI, without a second GUI lock."""
    if action == "refresh":
        return manager.list_profiles()
    if action not in {"install", "check", "select"}:
        raise ValueError("不明な操作です。")
    if not profile_id:
        raise ValueError("モデルを一覧から選んでください。")
    if action == "select":
        return manager.select(profile_id)
    return getattr(manager, action)(profile_id, cancel=cancel, progress=progress)


def present_model_terms(window, status: dict) -> bool:
    """Show complete documents on the Tk thread; the consent box starts unchecked."""
    import tkinter as tk
    from tkinter import ttk
    dialog = tk.Toplevel(window)
    dialog.title("Gemma3の利用条件")
    dialog.geometry("1000x700")
    dialog.minsize(760, 520)
    dialog.transient(window)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(1, weight=1)
    result = {"accepted": False}
    frame = ttk.Frame(dialog, padding=16)
    frame.grid(row=0, column=0, sticky="ew")
    ttk.Label(frame, text="Gemma3を導入・使用する前に、次の規約と禁止用途を確認してください。",
              wraplength=950).grid(row=0, column=0, sticky="w")
    ttk.Label(frame, text=f"採用した規約の版：{status['terms_version']}").grid(row=1, column=0, sticky="w", pady=(6, 0))
    notebook = ttk.Notebook(dialog)
    notebook.grid(row=1, column=0, sticky="nsew", padx=16)
    for document in status["documents"]:
        page = ttk.Frame(notebook, padding=8)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        notebook.add(page, text=document["title"])
        ttk.Label(page, text=document["url"]).grid(row=0, column=0, sticky="w", pady=(0, 8))
        text = tk.Text(page, wrap="word", font=("Yu Gothic UI", 10))
        text.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(page, orient="vertical", command=text.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        text.insert("1.0", document["text"])
        text.configure(state="disabled")
    footer = ttk.Frame(dialog, padding=16)
    footer.grid(row=2, column=0, sticky="ew")
    choice = tk.BooleanVar(master=dialog, value=False)
    checkbox = ttk.Checkbutton(footer, variable=choice,
        text="Gemma Terms of UseとGemma Prohibited Use Policyを確認し、これらの条件に同意します。")
    checkbox.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
    def accept():
        if choice.get():
            result["accepted"] = True
            dialog.destroy()
    accept_button = ttk.Button(footer, text="同意して続ける", command=accept, state="disabled")
    accept_button.grid(row=1, column=0, sticky="w")
    ttk.Button(footer, text="取り消す", command=dialog.destroy).grid(row=1, column=1, sticky="w", padx=12)
    choice.trace_add("write", lambda *_: accept_button.configure(state="normal" if choice.get() else "disabled"))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.grab_set()
    dialog.wait_window()
    return result["accepted"]


def confirm_model_terms(window, manager, profile_id: str) -> bool:
    status = manager.terms_status(profile_id)
    if not status["required"] or status["accepted"]:
        return True
    if not present_model_terms(window, status):
        return False
    # Hashes are taken from exactly the displayed documents and rechecked on save.
    manager.accept_terms(profile_id, {item["path"]: item["sha256"] for item in status["documents"]}, accepted=True)
    return True


class OperationWorker:
    """Keep cancellation and completion ordered until the UI receives completion."""

    def __init__(self):
        self.events: Queue[dict] = Queue()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active = False

    def start(self, action: str, operation: Callable):
        if self.active or (self.thread is not None and self.thread.is_alive()):
            raise RuntimeError("処理が実行中です。")
        self.cancel_event = threading.Event()
        self.active = True

        def progress(value):
            self.events.put({"kind": "progress", "action": action,
                             "value": dict(value) if isinstance(value, dict) else {"status": str(value)}})

        def run():
            completed = {"kind": "complete", "action": action}
            try:
                completed["result"] = operation(self.cancel_event, progress)
            except Exception as error:
                completed.update(error=str(error), code=getattr(error, "code", None),
                                 traceback=traceback.format_exc())
            self.events.put(completed)

        self.thread = threading.Thread(target=run, name="model-manager-operation", daemon=False)
        self.thread.start()

    def request_cancel(self):
        if self.active:
            self.cancel_event.set()

    def read_events(self):
        events = []
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            events.append(event)
            if event["kind"] == "complete":
                self.active = False
        return events

    @property
    def finished(self):
        return not self.active and (self.thread is None or not self.thread.is_alive())


class ModelManagerWindow:
    def __init__(self, window, root: Path, manager):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.window = window
        self.root = Path(root).resolve()
        self.manager = manager
        self.worker = OperationWorker()
        self.profiles: dict[str, dict] = {}
        self.closing = False
        self._refresh_pending = False
        self._cancel_allowed = False
        window.title("図面項目抽出 — モデル管理")
        window.geometry("980x580")
        window.minsize(800, 500)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        frame = ttk.Frame(window, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=3)
        frame.rowconfigure(7, weight=2)
        ttk.Label(frame, text="抽出に使うモデル", font=("Yu Gothic UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        self.selected_text = tk.StringVar(value="現在の抽出モデル：確認中")
        ttk.Label(frame, textvariable=self.selected_text).grid(row=1, column=0, sticky="w", pady=(8, 4))
        ttk.Label(frame, text="モデルを導入し、動作確認してから使用モデルに設定します。動作確認は抽出精度の保証ではありません。",
                  wraplength=920).grid(row=2, column=0, sticky="w", pady=(0, 10))

        table = ttk.Frame(frame)
        table.grid(row=3, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("profile", "model", "size", "installed", "checked", "selected")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=5)
        headings = ("モデル", "Ollamaモデル名", "容量", "導入", "動作確認", "抽出に使用")
        widths = (160, 235, 95, 70, 135, 85)
        for column, heading, width in zip(columns, headings, widths):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, minwidth=65, stretch=column in ("profile", "model", "checked"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _: self._update_buttons())

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, sticky="w", pady=12)
        self.buttons = {}
        for index, action in enumerate(("refresh", "install", "check", "select")):
            button = ttk.Button(buttons, text=BUTTON_LABELS[action], command=lambda action=action: self.start(action))
            button.grid(row=0, column=index, padx=(0, 8))
            self.buttons[action] = button
        self.cancel_button = ttk.Button(buttons, text="中断", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=0, column=4, padx=(8, 0))
        self.status_text = tk.StringVar(value="モデル一覧を確認します。")
        ttk.Label(frame, textvariable=self.status_text, wraplength=920).grid(row=5, column=0, sticky="w")
        self.progress = ttk.Progressbar(frame, maximum=100, mode="indeterminate")
        self.progress.grid(row=6, column=0, sticky="ew", pady=(6, 10))
        details = ttk.Frame(frame)
        details.grid(row=7, column=0, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(0, weight=1)
        self.history = tk.Text(details, height=6, wrap="word", state="disabled", font=("Yu Gothic UI", 10))
        self.history.grid(row=0, column=0, sticky="nsew")
        history_scroll = ttk.Scrollbar(details, orient="vertical", command=self.history.yview)
        history_scroll.grid(row=0, column=1, sticky="ns")
        self.history.configure(yscrollcommand=history_scroll.set)
        ttk.Label(frame, text="保存先はこの候補フォルダー内です。PDF処理は既存の前処理・抽出BATから実行できます。",
                  wraplength=920).grid(row=8, column=0, sticky="w", pady=(10, 0))
        self.start("refresh")
        window.after(80, self.poll)

    def _chosen_id(self):
        chosen = self.tree.selection()
        return chosen[0] if chosen else None

    def _update_buttons(self):
        idle = self.worker.finished and not self.closing
        chosen = self.profiles.get(self._chosen_id(), {})
        for action, button in self.buttons.items():
            enabled = idle and (action == "refresh" or bool(chosen))
            if action == "check":
                enabled = enabled and bool(chosen.get("installed"))
            if action == "select":
                enabled = enabled and bool(chosen.get("installed")) and bool(chosen.get("checked"))
            button.configure(state="normal" if enabled else "disabled")
        self.cancel_button.configure(state="normal" if self.worker.active and self._cancel_allowed
                                     and not self.worker.cancel_event.is_set() and not self.closing else "disabled")

    def _append_history(self, message: str):
        self.history.configure(state="normal")
        self.history.insert("end", f"{datetime.now():%H:%M:%S}  {message}\n")
        self.history.see("end")
        self.history.configure(state="disabled")

    def _show_profiles(self, profiles):
        previous = self._chosen_id()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.profiles = {p["profile_id"]: p for p in profiles}
        current = None
        for key, profile in self.profiles.items():
            label = profile.get("label", key)
            self.tree.insert("", "end", iid=key, values=(label, profile.get("model_tag", ""),
                format_size(profile.get("size_bytes")), "導入済み" if profile.get("installed") else "未導入",
                profile.get("check_status", "未確認"), "使用中" if profile.get("selected") else ""))
            if profile.get("selected"):
                current = key
                self.selected_text.set(f"現在の抽出モデル：{label}（{profile.get('model_tag', '')}）")
        if current is None:
            self.selected_text.set("現在の抽出モデル：未選択")
        chosen = previous if previous in self.profiles else current or next(iter(self.profiles), None)
        if chosen:
            self.tree.selection_set(chosen)
            self.tree.focus(chosen)

    def start(self, action: str):
        if not self.worker.finished or self.closing:
            return
        chosen = self._chosen_id()
        if action != "refresh" and chosen not in self.profiles:
            self.status_text.set("モデルを一覧から選んでください。")
            return
        if action != "refresh":
            try:
                if not confirm_model_terms(self.window, self.manager, chosen):
                    self.status_text.set("利用条件への同意を取り消しました。処理は開始していません。")
                    return
            except Exception as exc:
                message = f"利用条件を確認できません：{exc}"
                self.status_text.set(message)
                self._append_history(message)
                return
        target = self.profiles.get(chosen, {}).get("label", chosen or "")
        self._cancel_allowed = action in {"install", "check"}
        running = {"refresh": "モデルの状態を確認しています…", "install": f"{target}を導入しています…",
                   "check": f"{target}の動作を確認しています…", "select": f"{target}を抽出モデルに設定しています…"}
        started = "モデルの状態更新を開始しました。" if action == "refresh" else (
            f"{target}を抽出モデルに設定します。" if action == "select" else f"{target}の{ACTION_LABELS[action]}を開始しました。")
        self.status_text.set(running[action])
        self._append_history(started)
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.worker.start(action, lambda cancel, progress: perform_action(self.manager, action, chosen, cancel, progress))
        self._update_buttons()

    def cancel(self):
        if self.worker.active and self._cancel_allowed:
            self.worker.request_cancel()
            self.status_text.set("中断しています。保存とサーバーの終了を待っています…")
            self._update_buttons()

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.worker.request_cancel()
        self._refresh_pending = False
        self.status_text.set("終了しています。実行中の処理とサーバーの終了を待っています…")
        self._update_buttons()
        if self.worker.finished:
            self.window.destroy()

    def _handle_progress(self, value):
        if self.closing or self.worker.cancel_event.is_set():
            return
        completed, total = value.get("completed"), value.get("total")
        text = progress_message(value)
        if isinstance(total, (int, float)) and total > 0 and isinstance(completed, (int, float)):
            self.progress.stop()
            self.progress.configure(mode="determinate", value=max(0, min(100, completed * 100 / total)))
            text += f"  {format_size(completed)} / {format_size(total)}"
        elif str(self.progress.cget("mode")) != "indeterminate":
            self.progress.stop()
            self.progress.configure(mode="indeterminate", value=0)
            self.progress.start(12)
        self.status_text.set(text or "処理中…")

    def _handle_complete(self, event):
        self.progress.stop()
        self.progress.configure(value=0)
        action = event["action"]
        if "error" in event:
            if event.get("code") == "CANCELLED":
                message = "中断しました。保存済みモデルと結果は残しています。"
            else:
                message = f"{ACTION_LABELS[action]}ができません：{event['error']}"
            self._append_history(message)
            if not self.closing:
                self.status_text.set(message)
        elif action == "refresh":
            self._show_profiles(event["result"])
            if not self.closing:
                self.status_text.set("モデルの状態を更新しました。")
        else:
            result = event.get("result", {})
            message = f"{ACTION_LABELS[action]}が完了しました。"
            if action == "select":
                message += " 次の抽出から使用します。"
                model = result.get("model_tag", "") if isinstance(result, dict) else ""
                self.selected_text.set(f"現在の抽出モデル：{model}")
            self._append_history(message)
            if not self.closing:
                self.status_text.set(message)
        if action != "refresh" and not self.closing:
            self._refresh_pending = True

    def poll(self):
        for event in self.worker.read_events():
            if event["kind"] == "progress":
                self._handle_progress(event["value"])
            else:
                self._handle_complete(event)
        if self.closing and self.worker.finished:
            self.window.destroy()
            return
        if self._refresh_pending and self.worker.finished:
            self._refresh_pending = False
            self.start("refresh")
        self._update_buttons()
        self.window.after(80, self.poll)


def launch(root: Path, manager=None):
    """Open the manager window and preserve a diagnostic if startup fails."""
    root = Path(root).resolve()
    window = None
    try:
        # The process owns this setting; it does not change a user-wide preference.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
        import tkinter as tk
        if manager is None:
            from .model_manager import ModelManager
            manager = ModelManager(root)
        window = tk.Tk()
        ModelManagerWindow(window, root, manager)
        window.mainloop()
        return 0
    except Exception:
        diagnostic = traceback.format_exc()
        folder = root / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"model-gui-startup-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
        path.write_text(diagnostic, encoding="utf-8")
        if sys.stderr is not None:
            print(diagnostic, file=sys.stderr)
        if window is not None:
            try:
                window.destroy()
            except Exception:
                pass
        message = f"モデル管理を起動できませんでした。\n診断記録：{path}"
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "モデル管理", 0x10)
        except (AttributeError, OSError):
            pass
        return 1
