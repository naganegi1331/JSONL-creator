# -*- coding: utf-8 -*-
"""SQLite接続・スキーマ管理."""

import datetime
import sqlite3

from config import get_db_path


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


def load_existing_keys(conn):
    """DB内の全レコードを (instruction, input, output) のキー集合として返す.

    インポート時の完全一致重複チェックに用いる。
    """
    rows = conn.execute(
        "SELECT instruction, input, output FROM records"
    ).fetchall()
    return {(r["instruction"], r["input"], r["output"]) for r in rows}
