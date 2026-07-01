# -*- coding: utf-8 -*-
"""jsonl_io.py のインポート（重複・不正・類似重複・レガシー取り込み）のテスト.

Ollamaへの実通信は jsonl_io.ollama_embed をモックして再現する。
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import embeddings
import jsonl_io


DIM = 4  # NearDuplicateTest用の索引次元数（本番はconfig.EMBEDDING_DIM=768）


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def write_jsonl(lines):
    """与えた文字列行をJSONLファイルに書き出し、パスを返す（呼び出し側で削除）."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def record_line(instruction, input_value, output, source=None):
    obj = {"instruction": instruction, "input": input_value, "output": output}
    if source is not None:
        obj["source"] = source
    return json.dumps(obj, ensure_ascii=False)


# Ollamaが起動していない状況を再現（接続拒否で OSError）
def _ollama_unavailable(_text):
    raise OSError("Ollama not running")


class ImportBasicTest(unittest.TestCase):
    def setUp(self):
        self.conn = make_conn()
        db.init_db(self.conn)
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            if os.path.exists(p):
                os.remove(p)

    def _write(self, lines):
        path = write_jsonl(lines)
        self.paths.append(path)
        return path

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_adds_valid_records(self, _mock):
        path = self._write(
            [
                record_line("質問1", "", "回答1"),
                record_line("質問2", "前提2", "回答2", "出典2"),
            ]
        )
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["duplicate"], 0)
        self.assertEqual(result["invalid"], 0)
        count = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        self.assertEqual(count, 2)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_skips_exact_triple_duplicate_against_db(self, _mock):
        ts = db.now_iso()
        self.conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("質問1", "", "回答1", "既存出典", ts, ts),
        )
        self.conn.commit()

        path = self._write([record_line("質問1", "", "回答1", "新しい出典")])
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicate"], 1)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_skips_duplicate_within_file(self, _mock):
        path = self._write(
            [
                record_line("質問1", "", "回答1"),
                record_line("質問1", "", "回答1", "別出典"),
            ]
        )
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicate"], 1)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_same_question_different_output_is_not_duplicate(self, _mock):
        path = self._write(
            [
                record_line("質問1", "", "回答A"),
                record_line("質問1", "", "回答B"),
            ]
        )
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["duplicate"], 0)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_skips_invalid_rows(self, _mock):
        path = self._write(
            [
                "{壊れたJSON",              # JSONとして解析不可
                json.dumps([1, 2, 3]),       # dictでない
                record_line("", "", "回答"),  # instruction欠落
                record_line("質問", "", "   "),  # output空白のみ
                json.dumps({"instruction": 1, "input": "", "output": "回答"}),  # 型不正
                record_line("有効な質問", "", "有効な回答"),  # 1件だけ有効
            ]
        )
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["invalid"], 5)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_missing_source_key_imported_as_empty(self, _mock):
        path = self._write([record_line("質問", "", "回答")])  # sourceキーなし
        jsonl_io.import_jsonl_into_db(self.conn, path)
        row = self.conn.execute("SELECT source FROM records").fetchone()
        self.assertEqual(row["source"], "")

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_near_duplicate_stays_zero_without_ollama(self, _mock):
        path = self._write([record_line("質問", "", "回答")])
        result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["near_duplicate"], 0)


class NearDuplicateTest(unittest.TestCase):
    """Ollama利用可能時の類似重複検出を、固定ベクトルのモックで検証する."""

    def setUp(self):
        self.conn = make_conn()
        db.init_db(self.conn, dim=DIM)
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            if os.path.exists(p):
                os.remove(p)

    def _write(self, lines):
        path = write_jsonl(lines)
        self.paths.append(path)
        return path

    def _insert_with_embedding(self, instruction, output, vector):
        ts = db.now_iso()
        cur = self.conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (instruction, "", output, "", embeddings.serialize_embedding(vector), ts, ts),
        )
        self.conn.execute(
            "INSERT INTO vec_records(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, embeddings.serialize_embedding(vector)),
        )
        self.conn.commit()

    def test_detects_near_duplicate_when_similarity_high(self):
        # 既存ベクトルと、取り込む行のベクトルがほぼ同一 → 類似重複として計上
        self._insert_with_embedding("元の質問", "元の回答", [1.0, 0.0, 0.0, 0.0])
        path = self._write([record_line("言い回し違いの質問", "", "別の回答")])
        with mock.patch("jsonl_io.ollama_embed", return_value=[1.0, 0.0, 0.0, 0.0]):
            result = jsonl_io.import_jsonl_into_db(self.conn, path)
        # 完全一致ではないので追加はされ、件数のみ報告される
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["near_duplicate"], 1)

    def test_no_near_duplicate_when_similarity_low(self):
        self._insert_with_embedding("元の質問", "元の回答", [1.0, 0.0, 0.0, 0.0])
        path = self._write([record_line("全く別の質問", "", "全く別の回答")])
        with mock.patch("jsonl_io.ollama_embed", return_value=[0.0, 1.0, 0.0, 0.0]):
            result = jsonl_io.import_jsonl_into_db(self.conn, path)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["near_duplicate"], 0)

    def test_imported_record_stores_embedding(self):
        path = self._write([record_line("新しい質問", "", "新しい回答")])
        with mock.patch("jsonl_io.ollama_embed", return_value=[0.5, 0.5, 0.0, 0.0]):
            jsonl_io.import_jsonl_into_db(self.conn, path)
        row = self.conn.execute("SELECT embedding FROM records").fetchone()
        self.assertEqual(
            embeddings.deserialize_embedding(row["embedding"]), [0.5, 0.5, 0.0, 0.0]
        )
        indexed = self.conn.execute("SELECT COUNT(*) FROM vec_records").fetchone()[0]
        self.assertEqual(indexed, 1)


class LegacyImportTest(unittest.TestCase):
    def setUp(self):
        self.conn = make_conn()
        db.init_db(self.conn)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_returns_zero_when_file_missing(self, _mock):
        with mock.patch(
            "jsonl_io.get_export_path",
            return_value=os.path.join(tempfile.gettempdir(), "does_not_exist_xyz.jsonl"),
        ):
            self.assertEqual(jsonl_io.import_legacy_jsonl(self.conn), 0)

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_imports_existing_file(self, _mock):
        path = write_jsonl(
            [record_line("質問1", "", "回答1"), record_line("質問2", "", "回答2")]
        )
        try:
            with mock.patch("jsonl_io.get_export_path", return_value=path):
                added = jsonl_io.import_legacy_jsonl(self.conn)
            self.assertEqual(added, 2)
        finally:
            os.remove(path)


class RoundTripTest(unittest.TestCase):
    """DBへ保存 → エクスポート相当の書き出し → 再インポートで重複検出される流れ."""

    @mock.patch("jsonl_io.ollama_embed", side_effect=_ollama_unavailable)
    def test_export_then_reimport_is_deduplicated(self, _mock):
        conn = make_conn()
        db.init_db(conn)
        ts = db.now_iso()
        conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("質問", "前提", "回答", "出典", ts, ts),
        )
        conn.commit()

        # export_jsonl と同じ整形でJSONLを書き出す
        rows = conn.execute(
            "SELECT instruction, input, output, source FROM records ORDER BY id"
        ).fetchall()
        path = write_jsonl(
            [
                json.dumps(
                    {
                        "instruction": r["instruction"],
                        "input": r["input"],
                        "output": r["output"],
                        "source": r["source"],
                    },
                    ensure_ascii=False,
                )
                for r in rows
            ]
        )
        try:
            result = jsonl_io.import_jsonl_into_db(conn, path)
            self.assertEqual(result["added"], 0)
            self.assertEqual(result["duplicate"], 1)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
