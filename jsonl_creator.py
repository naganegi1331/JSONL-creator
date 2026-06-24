# -*- coding: utf-8 -*-
"""LLM学習データ蓄積GUIツール.

サポート業務で得たナレッジやトラブルシューティングの記録を、エンジニアが
直感的なGUIから入力し、LLMファインチューニング用のデータ（Alpaca形式）
として蓄積する。データはSQLiteで管理し、JSONL形式でエクスポートできる。

対象OS : Windows 10 / 11
GUI    : Tkinter（Python標準ライブラリ）
DB     : SQLite（sqlite3・標準ライブラリ）
"""

import datetime
import json
import os
import sqlite3
import sys
import tkinter as tk
from tkinter import messagebox, ttk


DB_FILENAME = "training_data.db"
EXPORT_FILENAME = "training_data.jsonl"


def get_base_dir():
    """基準ディレクトリ（DB・エクスポート先の親）を返す.

    exe実行時（PyInstaller等でfrozen）とスクリプト実行時の両方で、
    実行ファイル/スクリプトと同じディレクトリを正しく解決する。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_db_path():
    """SQLite DBファイルのフルパスを返す."""
    return os.path.join(get_base_dir(), DB_FILENAME)


def get_export_path():
    """エクスポート先 training_data.jsonl のフルパスを返す."""
    return os.path.join(get_base_dir(), EXPORT_FILENAME)


def connect_db():
    """DBへ接続し、行をdictライクに扱える接続を返す."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    """recordsテーブルが無ければ作成する."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            instruction TEXT NOT NULL,
            input       TEXT NOT NULL DEFAULT '',
            output      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    conn.commit()


def now_iso():
    """秒精度のISO形式タイムスタンプ文字列を返す."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def import_legacy_jsonl(conn):
    """既存の training_data.jsonl をDBへ取り込む（初回のみ呼ばれる想定）.

    取り込んだ件数を返す。ファイルが無い場合や解析に失敗した行は
    スキップする。
    """
    path = get_export_path()
    if not os.path.exists(path):
        return 0

    ts = now_iso()
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conn.execute(
                    "INSERT INTO records "
                    "(instruction, input, output, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        obj.get("instruction", ""),
                        obj.get("input", ""),
                        obj.get("output", ""),
                        ts,
                        ts,
                    ),
                )
                count += 1
        conn.commit()
    except OSError:
        # 取り込みは任意処理。読み込めなければ何もしない。
        return 0
    return count


def _preview(text, limit=40):
    """一覧表示用に、改行を空白へ畳んで短く切り詰めた文字列を返す."""
    flat = " ".join(text.splitlines())
    if len(flat) > limit:
        return flat[:limit] + "…"
    return flat


class JsonlCreatorApp:
    """入力フォーム・一覧・編集・削除・エクスポートを提供するアプリ."""

    def __init__(self, root, conn):
        self.root = root
        self.conn = conn
        self.current_id = None  # 編集中レコードのID（Noneなら新規入力）

        self.root.title("LLM学習データ蓄積ツール")
        self.root.geometry("950x600")
        self.root.minsize(700, 450)

        self.status_var = tk.StringVar()
        self._build_widgets()
        self._reload_list()
        self._update_status()

    # ------------------------------------------------------------------
    # ウィジェット構築
    # ------------------------------------------------------------------
    def _build_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        paned.add(self._build_list_pane(paned), weight=1)
        paned.add(self._build_form_pane(paned), weight=2)

    def _build_list_pane(self, parent):
        """左ペイン：保存済みデータの一覧."""
        frame = ttk.Frame(parent, padding=(8, 8))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="保存済みデータ（クリックで編集）").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        self.tree = ttk.Treeview(
            frame,
            columns=("id", "instruction", "output"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("instruction", text="Instruction")
        self.tree.heading("output", text="Output")
        self.tree.column("id", width=40, anchor="center", stretch=False)
        self.tree.column("instruction", width=160)
        self.tree.column("output", width=160)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=1, column=1, sticky="ns")

        return frame

    def _build_form_pane(self, parent):
        """右ペイン：入力フォームと操作ボタン."""
        frame = ttk.Frame(parent, padding=(8, 8))
        frame.columnconfigure(0, weight=1)

        # Instruction（必須）
        frame.rowconfigure(1, weight=1)
        ttk.Label(
            frame, text="Instruction (Q: 顧客からの質問や発生した課題)"
        ).grid(row=0, column=0, sticky="ew")
        self.instruction_text = tk.Text(frame, height=5, wrap="word")
        self.instruction_text.grid(row=1, column=0, sticky="nsew", pady=(0, 6))

        # Input（任意）
        frame.rowconfigure(3, weight=1)
        ttk.Label(
            frame, text="Input (Context: 前提条件、環境、エラーコードなど)"
        ).grid(row=2, column=0, sticky="ew")
        self.input_text = tk.Text(frame, height=5, wrap="word")
        self.input_text.grid(row=3, column=0, sticky="nsew", pady=(0, 6))

        # Output（必須）
        frame.rowconfigure(5, weight=1)
        ttk.Label(
            frame, text="Output (A: 解決策、実際の対応内容)"
        ).grid(row=4, column=0, sticky="ew")
        self.output_text = tk.Text(frame, height=5, wrap="word")
        self.output_text.grid(row=5, column=0, sticky="nsew", pady=(0, 6))

        # 編集状態の表示
        ttk.Label(frame, textvariable=self.status_var, foreground="#555").grid(
            row=6, column=0, sticky="w", pady=(0, 4)
        )

        # 操作ボタン
        btns = ttk.Frame(frame)
        btns.grid(row=7, column=0, sticky="ew")
        for i in range(4):
            btns.columnconfigure(i, weight=1)

        self.save_button = ttk.Button(
            btns, text="保存", command=self.save_entry
        )
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btns, text="新規入力", command=self.new_entry).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(btns, text="選択行を削除", command=self.delete_entry).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(
            btns, text="JSONLにエクスポート", command=self.export_jsonl
        ).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        return frame

    # ------------------------------------------------------------------
    # テキスト入出力ヘルパ
    # ------------------------------------------------------------------
    def _get_value(self, widget):
        """テキストエリアの値を取得し、末尾の改行を除去して返す."""
        return widget.get("1.0", "end").rstrip("\n")

    def _set_text(self, widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------
    def save_entry(self):
        """入力をバリデーションし、新規追加または更新を行う."""
        instruction = self._get_value(self.instruction_text)
        input_value = self._get_value(self.input_text)
        output = self._get_value(self.output_text)

        # 必須入力チェック（空白文字のみも空欄とみなす）
        if not instruction.strip() or not output.strip():
            messagebox.showwarning("入力エラー", "必須項目が入力されていません")
            return

        ts = now_iso()
        try:
            if self.current_id is None:
                self.conn.execute(
                    "INSERT INTO records "
                    "(instruction, input, output, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (instruction, input_value, output, ts, ts),
                )
            else:
                self.conn.execute(
                    "UPDATE records SET instruction = ?, input = ?, "
                    "output = ?, updated_at = ? WHERE id = ?",
                    (instruction, input_value, output, ts, self.current_id),
                )
            self.conn.commit()
        except sqlite3.Error as e:
            # 書き込み権限なし、DBロック、ディスク障害などで保存に失敗した場合。
            # 入力データは画面上に保持したまま処理を中断する。
            messagebox.showerror(
                "保存エラー",
                "データベースへの保存に失敗しました。\n\n"
                f"詳細: {e}",
            )
            return

        messagebox.showinfo("保存完了", "データを保存しました。")
        self._clear_inputs()
        self._reload_list()

    def new_entry(self):
        """フォームをクリアして新規入力モードに戻す."""
        self._clear_inputs()

    def delete_entry(self):
        """一覧で選択中のレコードを削除する."""
        if self.current_id is None:
            messagebox.showwarning(
                "削除", "削除する行を一覧から選択してください。"
            )
            return
        if not messagebox.askyesno(
            "削除確認", "選択したデータを削除します。よろしいですか？"
        ):
            return
        try:
            self.conn.execute(
                "DELETE FROM records WHERE id = ?", (self.current_id,)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror(
                "削除エラー", f"削除に失敗しました。\n\n詳細: {e}"
            )
            return
        self._clear_inputs()
        self._reload_list()

    def export_jsonl(self):
        """DB内の全レコードをAlpaca形式のJSONLとして書き出す."""
        try:
            rows = self.conn.execute(
                "SELECT instruction, input, output FROM records ORDER BY id"
            ).fetchall()
        except sqlite3.Error as e:
            messagebox.showerror(
                "エクスポートエラー",
                f"データの読み込みに失敗しました。\n\n詳細: {e}",
            )
            return

        if not rows:
            messagebox.showwarning(
                "エクスポート", "エクスポートするデータがありません。"
            )
            return

        try:
            with open(get_export_path(), "w", encoding="utf-8") as f:
                for r in rows:
                    record = {
                        "instruction": r["instruction"],
                        "input": r["input"],
                        "output": r["output"],
                    }
                    # 日本語はエスケープせず、特殊文字は json.dumps で正しく処理
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            # 書き込み権限なし、ネットワーク切断などで保存に失敗した場合。
            messagebox.showerror(
                "エクスポートエラー",
                "ファイルの保存に失敗しました。\n"
                "書き込み権限やネットワーク接続を確認してください。\n\n"
                f"詳細: {e}",
            )
            return

        messagebox.showinfo(
            "エクスポート完了",
            f"{len(rows)} 件を {EXPORT_FILENAME} に書き出しました。",
        )

    # ------------------------------------------------------------------
    # 一覧・選択・状態
    # ------------------------------------------------------------------
    def _reload_list(self):
        """DBの内容で一覧を再描画する."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.conn.execute(
            "SELECT id, instruction, output FROM records ORDER BY id"
        ).fetchall()
        for r in rows:
            self.tree.insert(
                "",
                "end",
                iid=str(r["id"]),
                values=(
                    r["id"],
                    _preview(r["instruction"]),
                    _preview(r["output"]),
                ),
            )

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        self._load_record(int(sel[0]))

    def _load_record(self, rec_id):
        """選択レコードをフォームへ読み込み、編集モードにする."""
        row = self.conn.execute(
            "SELECT * FROM records WHERE id = ?", (rec_id,)
        ).fetchone()
        if row is None:
            return
        self.current_id = rec_id
        self._set_text(self.instruction_text, row["instruction"])
        self._set_text(self.input_text, row["input"])
        self._set_text(self.output_text, row["output"])
        self._update_status()

    def _clear_inputs(self):
        """フォームをクリアし、新規入力モードへ戻す."""
        self.current_id = None
        for w in (self.instruction_text, self.input_text, self.output_text):
            w.delete("1.0", "end")
        selection = self.tree.selection()
        if selection:
            self.tree.selection_remove(selection)
        self._update_status()
        self.instruction_text.focus_set()

    def _update_status(self):
        """編集状態の表示と保存ボタンのラベルを更新する."""
        if self.current_id is None:
            self.status_var.set("● 新規入力中")
            self.save_button.config(text="保存（新規追加）")
        else:
            self.status_var.set(f"● ID {self.current_id} を編集中")
            self.save_button.config(text="保存（更新）")


def main():
    db_existed = os.path.exists(get_db_path())
    conn = connect_db()
    init_db(conn)

    imported = 0
    if not db_existed:
        # DB新規作成時のみ、既存の training_data.jsonl を取り込む
        imported = import_legacy_jsonl(conn)

    root = tk.Tk()
    JsonlCreatorApp(root, conn)
    if imported:
        messagebox.showinfo(
            "インポート",
            f"既存の {EXPORT_FILENAME} から {imported} 件を取り込みました。",
        )
    try:
        root.mainloop()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
