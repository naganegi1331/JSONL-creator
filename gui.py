# -*- coding: utf-8 -*-
"""入力フォーム・一覧・編集・削除・インポート/エクスポート・類似検索のGUI."""

import json
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import EXPORT_FILENAME, get_export_path
from db import now_iso
from embeddings import (
    cosine_similarity,
    deserialize_embedding,
    embedding_text,
    ollama_embed,
    serialize_embedding,
)
from jsonl_io import import_jsonl_into_db


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
        """左ペイン：類似検索と保存済みデータの一覧."""
        frame = ttk.Frame(parent, padding=(8, 8))
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        # 類似検索（Ollamaの埋め込みでベクトル化済みデータの中から検索）
        search_frame = ttk.Frame(frame)
        search_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        search_frame.columnconfigure(0, weight=1)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        search_entry.bind("<Return>", lambda event: self.search_similar())
        ttk.Button(
            search_frame, text="類似検索", command=self.search_similar
        ).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(
            search_frame, text="一覧に戻る", command=self._reload_list
        ).grid(row=0, column=2)

        self.list_label_var = tk.StringVar(
            value="保存済みデータ（クリックで編集 / Ctrl・Shiftで複数選択）"
        )
        ttk.Label(frame, textvariable=self.list_label_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        self.tree = ttk.Treeview(
            frame,
            columns=("id", "instruction", "output", "score"),
            show="headings",
            selectmode="extended",
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("instruction", text="Instruction")
        self.tree.heading("output", text="Output")
        self.tree.heading("score", text="類似度")
        self.tree.column("id", width=40, anchor="center", stretch=False)
        self.tree.column("instruction", width=160)
        self.tree.column("output", width=140)
        self.tree.column("score", width=55, anchor="center", stretch=False)
        self.tree.grid(row=2, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=2, column=1, sticky="ns")

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

        # Source（任意）：引用元のURLや社内ユーザーガイドの参照先など
        ttk.Label(
            frame, text="Source (引用元: 参考URL、社内ユーザーガイド等／任意)"
        ).grid(row=6, column=0, sticky="ew")
        self.source_text = tk.Text(frame, height=2, wrap="word")
        self.source_text.grid(row=7, column=0, sticky="ew", pady=(0, 6))

        # 編集状態の表示
        ttk.Label(frame, textvariable=self.status_var, foreground="#555").grid(
            row=8, column=0, sticky="w", pady=(0, 4)
        )

        # 操作ボタン（2段構成：上段=レコード操作、下段=ファイル入出力）
        btns = ttk.Frame(frame)
        btns.grid(row=9, column=0, sticky="ew")
        for i in range(6):
            btns.columnconfigure(i, weight=1)

        self.save_button = ttk.Button(
            btns, text="保存", command=self.save_entry
        )
        self.save_button.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=(0, 3), pady=(0, 4)
        )
        ttk.Button(btns, text="新規入力", command=self.new_entry).grid(
            row=0, column=2, columnspan=2, sticky="ew", padx=3, pady=(0, 4)
        )
        ttk.Button(btns, text="選択行を削除", command=self.delete_entry).grid(
            row=0, column=4, columnspan=2, sticky="ew", padx=(3, 0), pady=(0, 4)
        )

        ttk.Button(
            btns, text="JSONLをインポート", command=self.import_jsonl
        ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 3))
        ttk.Button(
            btns, text="JSONLにエクスポート", command=self.export_jsonl
        ).grid(row=1, column=3, columnspan=3, sticky="ew", padx=(3, 0))

        ttk.Button(
            btns,
            text="ベクトル化（未処理分を一括処理・Ollama使用）",
            command=self.embed_pending,
        ).grid(row=2, column=0, columnspan=6, sticky="ew", pady=(4, 0))

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
        source = self._get_value(self.source_text)

        # 必須入力チェック（空白文字のみも空欄とみなす）
        if not instruction.strip() or not output.strip():
            messagebox.showwarning("入力エラー", "必須項目が入力されていません")
            return

        ts = now_iso()
        try:
            if self.current_id is None:
                self.conn.execute(
                    "INSERT INTO records "
                    "(instruction, input, output, source, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (instruction, input_value, output, source, ts, ts),
                )
            else:
                # 内容を更新した場合、既存の埋め込みベクトルは古くなるため
                # クリアする（再ベクトル化は「ベクトル化（未処理分）」で行う）
                self.conn.execute(
                    "UPDATE records SET instruction = ?, input = ?, "
                    "output = ?, source = ?, embedding = '', updated_at = ? "
                    "WHERE id = ?",
                    (instruction, input_value, output, source, ts, self.current_id),
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
        """一覧で選択中のレコード（複数可）を削除する."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning(
                "削除", "削除する行を一覧から選択してください。"
            )
            return

        count = len(selection)
        message = (
            f"選択した {count} 件のデータを削除します。よろしいですか？"
            if count > 1
            else "選択したデータを削除します。よろしいですか？"
        )
        if not messagebox.askyesno("削除確認", message):
            return

        ids = [(int(iid),) for iid in selection]
        try:
            self.conn.executemany("DELETE FROM records WHERE id = ?", ids)
            self.conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror(
                "削除エラー", f"削除に失敗しました。\n\n詳細: {e}"
            )
            return
        self._clear_inputs()
        self._reload_list()

    def import_jsonl(self):
        """JSONLファイルを選択してDBへ取り込む（完全一致重複はスキップ）."""
        path = filedialog.askopenfilename(
            title="インポートするJSONLファイルを選択",
            filetypes=[
                ("JSONL / JSON ファイル", "*.jsonl *.json"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not path:
            return  # キャンセル

        try:
            result = import_jsonl_into_db(self.conn, path)
        except OSError as e:
            messagebox.showerror(
                "インポートエラー",
                "ファイルの読み込みに失敗しました。\n\n" f"詳細: {e}",
            )
            return
        except sqlite3.Error as e:
            messagebox.showerror(
                "インポートエラー",
                "データベースへの保存に失敗しました。\n\n" f"詳細: {e}",
            )
            return

        self._reload_list()
        messagebox.showinfo(
            "インポート完了",
            f"追加: {result['added']} 件\n"
            f"重複スキップ: {result['duplicate']} 件\n"
            f"不正スキップ: {result['invalid']} 件\n"
            f"類似重複（要確認・追加済み）: {result['near_duplicate']} 件",
        )

    def export_jsonl(self):
        """DB内の全レコードをAlpaca形式のJSONLとして書き出す."""
        try:
            rows = self.conn.execute(
                "SELECT instruction, input, output, source "
                "FROM records ORDER BY id"
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
                    # 学習の三つ組はそのまま保ち、引用元は source 別キーで付与
                    record = {
                        "instruction": r["instruction"],
                        "input": r["input"],
                        "output": r["output"],
                        "source": r["source"],
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

    def embed_pending(self):
        """未ベクトル化のレコードをOllamaで一括ベクトル化する."""
        rows = self.conn.execute(
            "SELECT id, instruction, input FROM records WHERE embedding = ''"
        ).fetchall()
        if not rows:
            messagebox.showinfo("ベクトル化", "未処理のデータはありません。")
            return

        succeeded = 0
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            for i, r in enumerate(rows):
                text = embedding_text(r["instruction"], r["input"])
                try:
                    vector = ollama_embed(text)
                except (OSError, ValueError) as e:
                    if i == 0:
                        # 最初の1件で失敗した場合はOllamaに接続できないとみなす
                        messagebox.showerror(
                            "ベクトル化エラー",
                            "Ollamaに接続できませんでした。Ollamaが起動して"
                            "いて、埋め込みモデルが利用可能か確認してください。"
                            f"\n\n詳細: {e}",
                        )
                        return
                    # 途中で接続できなくなった場合は、そこまでの結果を保存して終了
                    break
                self.conn.execute(
                    "UPDATE records SET embedding = ? WHERE id = ?",
                    (serialize_embedding(vector), r["id"]),
                )
                succeeded += 1
            self.conn.commit()
        finally:
            self.root.config(cursor="")

        messagebox.showinfo(
            "ベクトル化完了",
            f"{succeeded} / {len(rows)} 件のベクトル化が完了しました。",
        )

    def search_similar(self):
        """検索ボックスのテキストをOllamaでベクトル化し、類似データを探す."""
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning(
                "類似検索", "検索したい質問やキーワードを入力してください。"
            )
            return

        try:
            query_vector = ollama_embed(query)
        except (OSError, ValueError) as e:
            messagebox.showerror(
                "類似検索エラー",
                "Ollamaに接続できませんでした。Ollamaが起動していて、"
                f"埋め込みモデルが利用可能か確認してください。\n\n詳細: {e}",
            )
            return

        rows = self.conn.execute(
            "SELECT id, instruction, output, embedding FROM records "
            "WHERE embedding != ''"
        ).fetchall()
        scored = []
        for r in rows:
            vector = deserialize_embedding(r["embedding"])
            if vector:
                scored.append((cosine_similarity(query_vector, vector), r))

        if not scored:
            messagebox.showinfo(
                "類似検索",
                "ベクトル化済みのデータがありません。先に"
                "「ベクトル化（未処理分を一括処理）」を実行してください。",
            )
            return

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:20]

        for item in self.tree.get_children():
            self.tree.delete(item)
        for score, r in top:
            self.tree.insert(
                "",
                "end",
                iid=str(r["id"]),
                values=(
                    r["id"],
                    _preview(r["instruction"]),
                    _preview(r["output"]),
                    f"{score:.3f}",
                ),
            )
        self.list_label_var.set(f"類似検索結果（上位{len(top)}件・クリックで編集）")

    # ------------------------------------------------------------------
    # 一覧・選択・状態
    # ------------------------------------------------------------------
    def _reload_list(self):
        """DBの内容で一覧を再描画する."""
        self.list_label_var.set(
            "保存済みデータ（クリックで編集 / Ctrl・Shiftで複数選択）"
        )
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
                    "",
                ),
            )

    def _on_select(self, event):
        sel = self.tree.selection()
        if len(sel) == 1:
            # 単一選択時のみフォームへ読み込んで編集モードにする
            self._load_record(int(sel[0]))
        else:
            # 0件または複数選択時は単一の編集対象を持たず、フォームを空にする
            self.current_id = None
            for w in (
                self.instruction_text,
                self.input_text,
                self.output_text,
                self.source_text,
            ):
                w.delete("1.0", "end")
            self._update_status()

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
        self._set_text(self.source_text, row["source"])
        self._update_status()

    def _clear_inputs(self):
        """フォームをクリアし、新規入力モードへ戻す."""
        self.current_id = None
        for w in (
            self.instruction_text,
            self.input_text,
            self.output_text,
            self.source_text,
        ):
            w.delete("1.0", "end")
        selection = self.tree.selection()
        if selection:
            self.tree.selection_remove(selection)
        self._update_status()
        self.instruction_text.focus_set()

    def _update_status(self):
        """編集状態の表示と保存ボタンのラベルを更新する."""
        if self.current_id is not None:
            self.status_var.set(f"● ID {self.current_id} を編集中")
            self.save_button.config(text="保存（更新）")
            return

        selected = len(self.tree.selection())
        if selected >= 2:
            self.status_var.set(f"● {selected} 件を選択中（削除できます）")
        else:
            self.status_var.set("● 新規入力中")
        self.save_button.config(text="保存（新規追加）")
