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
import math
import os
import sqlite3
import sys
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import filedialog, messagebox, ttk


DB_FILENAME = "training_data.db"
EXPORT_FILENAME = "training_data.jsonl"

# Ollama（ローカルLLM）の埋め込みAPI設定。
# 事前に `ollama pull nomic-embed-text` 等でモデルを取得し、
# Ollamaをローカルで起動しておく必要がある。
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT = 10  # 秒

# インポート時、この類似度（コサイン類似度）以上を「類似重複」として報告する。
NEAR_DUPLICATE_THRESHOLD = 0.93


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
    """recordsテーブルが無ければ作成し、必要に応じてマイグレーションする."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            instruction TEXT NOT NULL,
            input       TEXT NOT NULL DEFAULT '',
            output      TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    # 旧バージョンで作成されたDBへの列追加（マイグレーション）
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
    if "source" not in columns:
        conn.execute(
            "ALTER TABLE records ADD COLUMN source TEXT NOT NULL DEFAULT ''"
        )
    if "embedding" not in columns:
        conn.execute(
            "ALTER TABLE records ADD COLUMN embedding TEXT NOT NULL DEFAULT ''"
        )
    conn.commit()


def now_iso():
    """秒精度のISO形式タイムスタンプ文字列を返す."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def embedding_text(instruction, input_value):
    """埋め込み対象とするテキストを instruction と input から組み立てる."""
    instruction = instruction.strip()
    input_value = input_value.strip()
    if input_value:
        return instruction + "\n\n" + input_value
    return instruction


def ollama_embed(text, model=OLLAMA_EMBEDDING_MODEL, timeout=OLLAMA_TIMEOUT):
    """Ollamaのローカル埋め込みAPIを呼び出し、ベクトル（list[float]）を返す.

    Ollamaが起動していない、モデルが未取得などの場合は OSError
    （ConnectionRefusedError・urllib.error.URLError等を含む）を呼び出し元に
    伝える。レスポンスの形式が想定外の場合は ValueError を送出する。
    """
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    vector = body.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollamaから埋め込みベクトルを取得できませんでした。")
    return vector


def serialize_embedding(vector):
    """埋め込みベクトルをDB保存用のJSON文字列に変換する."""
    return json.dumps(vector)


def deserialize_embedding(text):
    """DBに保存されたJSON文字列を埋め込みベクトルに変換する.

    未設定（空文字）の場合は None を返す。
    """
    if not text:
        return None
    return json.loads(text)


def cosine_similarity(vector_a, vector_b):
    """2つのベクトルのコサイン類似度を返す（外部ライブラリ不使用）."""
    if not vector_a or not vector_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_existing_keys(conn):
    """DB内の全レコードを (instruction, input, output) のキー集合として返す.

    インポート時の完全一致重複チェックに用いる。
    """
    rows = conn.execute(
        "SELECT instruction, input, output FROM records"
    ).fetchall()
    return {(r["instruction"], r["input"], r["output"]) for r in rows}


def load_existing_embeddings(conn):
    """DB内のベクトル化済みレコードを [(id, vector), ...] として返す."""
    rows = conn.execute(
        "SELECT id, embedding FROM records WHERE embedding != ''"
    ).fetchall()
    result = []
    for r in rows:
        vector = deserialize_embedding(r["embedding"])
        if vector:
            result.append((r["id"], vector))
    return result


def import_jsonl_into_db(conn, path):
    """JSONLファイルをDBへ取り込む.

    instruction・input・output の3フィールドが完全一致するレコードは
    重複としてスキップする（DB内の既存データおよびファイル内の先行行の
    両方が対象）。必須項目（instruction / output）が空のもの、JSONとして
    解析できないもの、型が不正なものはスキップする。

    Ollamaが起動していれば、取り込む各レコードを埋め込みベクトル化して
    保存し、既存データとのコサイン類似度が NEAR_DUPLICATE_THRESHOLD 以上
    の場合は「類似重複」として報告する（スキップはしない。同じ質問でも
    回答が異なるデータを残すのと同じ考え方）。Ollamaに接続できない場合は
    ベクトル化を行わずに通常のインポートを続行する。

    戻り値: {"added": 追加件数, "duplicate": 重複件数, "invalid": 不正件数,
            "near_duplicate": 類似重複件数}
    """
    existing = load_existing_keys(conn)
    existing_embeddings = load_existing_embeddings(conn)
    added = duplicate = invalid = near_duplicate = 0
    ts = now_iso()
    ollama_available = True

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(obj, dict):
                invalid += 1
                continue

            instruction = obj.get("instruction", "")
            input_value = obj.get("input", "")
            output = obj.get("output", "")
            source = obj.get("source", "")
            if not (
                isinstance(instruction, str)
                and isinstance(input_value, str)
                and isinstance(output, str)
                and isinstance(source, str)
            ):
                invalid += 1
                continue
            # 必須項目チェック（空白文字のみも空欄とみなす）
            if not instruction.strip() or not output.strip():
                invalid += 1
                continue

            # 重複判定は学習の三つ組（instruction/input/output）で行う。
            # 同一の三つ組はsourceが異なっても重複として先勝ちでスキップする。
            key = (instruction, input_value, output)
            if key in existing:
                duplicate += 1
                continue

            vector = None
            if ollama_available:
                try:
                    vector = ollama_embed(embedding_text(instruction, input_value))
                except (OSError, ValueError):
                    # Ollamaに接続できないとみなし、以降は試行しない
                    ollama_available = False

            if vector is not None and existing_embeddings:
                best_score = max(
                    cosine_similarity(vector, v) for _, v in existing_embeddings
                )
                if best_score >= NEAR_DUPLICATE_THRESHOLD:
                    near_duplicate += 1

            cur = conn.execute(
                "INSERT INTO records "
                "(instruction, input, output, source, embedding, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    instruction,
                    input_value,
                    output,
                    source,
                    serialize_embedding(vector) if vector is not None else "",
                    ts,
                    ts,
                ),
            )
            existing.add(key)
            if vector is not None:
                existing_embeddings.append((cur.lastrowid, vector))
            added += 1

    conn.commit()
    return {
        "added": added,
        "duplicate": duplicate,
        "invalid": invalid,
        "near_duplicate": near_duplicate,
    }


def import_legacy_jsonl(conn):
    """既存の training_data.jsonl をDBへ取り込む（初回のみ呼ばれる想定）.

    取り込んだ件数を返す。ファイルが無い、または読み込めない場合は0。
    """
    path = get_export_path()
    if not os.path.exists(path):
        return 0
    try:
        result = import_jsonl_into_db(conn, path)
    except OSError:
        # 取り込みは任意処理。読み込めなければ何もしない。
        return 0
    return result["added"]


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
