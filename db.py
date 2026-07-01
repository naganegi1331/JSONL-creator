# -*- coding: utf-8 -*-
"""SQLite接続・スキーマ管理."""

import datetime
import json
import sqlite3

import sqlite_vec

from config import EMBEDDING_DIM, get_db_path


def connect_db():
    """DBへ接続し、行をdictライクに扱える接続を返す."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    load_vec_extension(conn)
    return conn


def load_vec_extension(conn):
    """sqlite-vec拡張（ベクトル検索用のvec0仮想テーブルを提供）を読み込む."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def init_db(conn, dim=EMBEDDING_DIM):
    """recordsテーブルが無ければ作成し、必要に応じてマイグレーションする.

    sqlite-vecのベクトル索引テーブル（vec_records）も無ければ作成し、
    旧バージョンのDBで records.embedding に保存済みだが索引が無いベクトルを
    バックフィルする。dim は索引テーブルの次元数（テスト用に差し替え可能）。
    """
    load_vec_extension(conn)
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

    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0("
        f"embedding float[{dim}] distance_metric=cosine)"
    )
    # 旧バージョンのDBで records.embedding には値があるが vec_records に
    # まだ登録されていない行を索引へ取り込む（sqlite-vec導入前のデータの移行）。
    # ノルム0のベクトルは索引に登録しない（embeddings.index_vectorと同じ理由）。
    rows = conn.execute(
        "SELECT id, embedding FROM records WHERE embedding != '' "
        "AND id NOT IN (SELECT rowid FROM vec_records)"
    ).fetchall()
    for row in rows:
        if any(component != 0 for component in json.loads(row["embedding"])):
            conn.execute(
                "INSERT INTO vec_records(rowid, embedding) VALUES (?, ?)",
                (row["id"], row["embedding"]),
            )
    conn.commit()


def now_iso():
    """秒精度のISO形式タイムスタンプ文字列を返す."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_existing_keys(conn):
    """DB内の全レコードを (instruction, input, output) のキー集合として返す.

    インポート時の完全一致重複チェックに用いる。
    """
    rows = conn.execute(
        "SELECT instruction, input, output FROM records"
    ).fetchall()
    return {(r["instruction"], r["input"], r["output"]) for r in rows}
