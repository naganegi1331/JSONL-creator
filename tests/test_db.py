# -*- coding: utf-8 -*-
"""db.py のスキーマ作成・マイグレーション・補助関数のテスト."""

import datetime
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db


def make_conn():
    """メモリ上のSQLite接続を返す（実DBファイルに触れない）."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class InitDbTest(unittest.TestCase):
    def test_creates_records_table_with_expected_columns(self):
        conn = make_conn()
        db.init_db(conn)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
        self.assertEqual(
            columns,
            {
                "id",
                "instruction",
                "input",
                "output",
                "source",
                "created_at",
                "updated_at",
                "embedding",
            },
        )

    def test_init_db_is_idempotent(self):
        conn = make_conn()
        db.init_db(conn)
        # 2回呼んでも例外なく通る（CREATE TABLE IF NOT EXISTS）
        db.init_db(conn)
        count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        self.assertEqual(count, 0)

    def test_migration_adds_missing_columns(self):
        # source / embedding 列を持たない旧スキーマを用意
        conn = make_conn()
        conn.execute(
            """
            CREATE TABLE records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT NOT NULL,
                input       TEXT NOT NULL DEFAULT '',
                output      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO records (instruction, input, output, created_at, updated_at) "
            "VALUES ('旧質問', '', '旧回答', '2020-01-01T00:00:00', '2020-01-01T00:00:00')"
        )
        conn.commit()

        db.init_db(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
        self.assertIn("source", columns)
        self.assertIn("embedding", columns)
        # 既存レコードは保持され、追加列は空文字で埋まる
        row = conn.execute("SELECT source, embedding FROM records").fetchone()
        self.assertEqual(row["source"], "")
        self.assertEqual(row["embedding"], "")


class NowIsoTest(unittest.TestCase):
    def test_now_iso_is_second_precision_and_parseable(self):
        value = db.now_iso()
        # マイクロ秒を含まない（秒精度）
        self.assertNotIn(".", value)
        # ISO形式として解釈できる
        parsed = datetime.datetime.fromisoformat(value)
        self.assertIsInstance(parsed, datetime.datetime)


class LoadExistingKeysTest(unittest.TestCase):
    def test_returns_triples_ignoring_source(self):
        conn = make_conn()
        db.init_db(conn)
        ts = db.now_iso()
        conn.executemany(
            "INSERT INTO records "
            "(instruction, input, output, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("質問A", "入力A", "回答A", "出典1", ts, ts),
                ("質問B", "", "回答B", "", ts, ts),
            ],
        )
        conn.commit()

        keys = db.load_existing_keys(conn)
        self.assertEqual(
            keys,
            {("質問A", "入力A", "回答A"), ("質問B", "", "回答B")},
        )

    def test_empty_db_returns_empty_set(self):
        conn = make_conn()
        db.init_db(conn)
        self.assertEqual(db.load_existing_keys(conn), set())


if __name__ == "__main__":
    unittest.main()
