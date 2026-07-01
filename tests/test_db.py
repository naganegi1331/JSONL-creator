# -*- coding: utf-8 -*-
"""db.py のスキーマ作成・マイグレーション・補助関数のテスト."""

import datetime
import json
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

DIM = 4  # テスト用の索引次元数（本番はconfig.EMBEDDING_DIM=768）


def make_conn():
    """メモリ上のSQLite接続を返す（実DBファイルに触れない）."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class InitDbTest(unittest.TestCase):
    def test_creates_records_table_with_expected_columns(self):
        conn = make_conn()
        db.init_db(conn, dim=DIM)
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
        db.init_db(conn, dim=DIM)
        # 2回呼んでも例外なく通る（CREATE TABLE IF NOT EXISTS）
        db.init_db(conn, dim=DIM)
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

        db.init_db(conn, dim=DIM)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
        self.assertIn("source", columns)
        self.assertIn("embedding", columns)
        # 既存レコードは保持され、追加列は空文字で埋まる
        row = conn.execute("SELECT source, embedding FROM records").fetchone()
        self.assertEqual(row["source"], "")
        self.assertEqual(row["embedding"], "")


class VecRecordsTest(unittest.TestCase):
    """sqlite-vecのベクトル索引テーブル（vec_records）の作成・移行のテスト."""

    def test_creates_vec_records_index_table(self):
        conn = make_conn()
        db.init_db(conn, dim=DIM)

        conn.execute(
            "INSERT INTO vec_records(rowid, embedding) VALUES (1, ?)",
            (json.dumps([1.0, 0.0, 0.0, 0.0]),),
        )
        row = conn.execute(
            "SELECT rowid, distance FROM vec_records "
            "WHERE embedding MATCH ? AND k = 1",
            (json.dumps([1.0, 0.0, 0.0, 0.0]),),
        ).fetchone()
        self.assertEqual(row["rowid"], 1)
        self.assertAlmostEqual(row["distance"], 0.0)

    def test_backfills_index_from_legacy_embedding_column(self):
        # sqlite-vec導入前のDB相当：records.embeddingに値はあるが
        # vec_recordsにはまだ登録されていない状態を再現する。
        conn = make_conn()
        db.init_db(conn, dim=DIM)
        ts = db.now_iso()
        cur = conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("質問", "", "回答", "", json.dumps([1.0, 0.0, 0.0, 0.0]), ts, ts),
        )
        conn.execute("DELETE FROM vec_records")
        conn.commit()
        rec_id = cur.lastrowid

        db.init_db(conn, dim=DIM)  # 再度呼び出すとバックフィルされる

        row = conn.execute(
            "SELECT rowid FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()
        self.assertIsNotNone(row)

    def test_does_not_backfill_already_indexed_rows(self):
        conn = make_conn()
        db.init_db(conn, dim=DIM)
        ts = db.now_iso()
        cur = conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("質問", "", "回答", "", json.dumps([1.0, 0.0, 0.0, 0.0]), ts, ts),
        )
        rec_id = cur.lastrowid
        conn.execute(
            "INSERT INTO vec_records(rowid, embedding) VALUES (?, ?)",
            (rec_id, json.dumps([0.0, 1.0, 0.0, 0.0])),
        )
        conn.commit()

        db.init_db(conn, dim=DIM)  # 既に索引済みなので上書きされない

        count = conn.execute(
            "SELECT COUNT(*) FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()[0]
        self.assertEqual(count, 1)


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
