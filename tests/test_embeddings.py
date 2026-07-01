# -*- coding: utf-8 -*-
"""embeddings.py のテキスト整形・直列化・sqlite-vecによる類似度計算のテスト.

Ollamaへの実通信は行わず、純粋関数とsqlite-vec索引を使ったDB操作のみを検証する。
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import embeddings

DIM = 4  # テスト用の索引次元数（本番はconfig.EMBEDDING_DIM=768）


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn, dim=DIM)
    return conn


def insert_record(conn, instruction="質問", output="回答"):
    """recordsに1行挿入し、そのidを返す（ベクトルは未設定）."""
    ts = db.now_iso()
    cur = conn.execute(
        "INSERT INTO records "
        "(instruction, input, output, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (instruction, "", output, "", ts, ts),
    )
    conn.commit()
    return cur.lastrowid


class EmbeddingTextTest(unittest.TestCase):
    def test_combines_instruction_and_input(self):
        self.assertEqual(
            embeddings.embedding_text("質問", "前提"), "質問\n\n前提"
        )

    def test_omits_empty_input(self):
        self.assertEqual(embeddings.embedding_text("質問", ""), "質問")
        self.assertEqual(embeddings.embedding_text("質問", "   "), "質問")

    def test_strips_whitespace(self):
        self.assertEqual(
            embeddings.embedding_text("  質問  ", "  前提  "), "質問\n\n前提"
        )


class SerializeTest(unittest.TestCase):
    def test_round_trip(self):
        vector = [0.1, 0.2, -0.3]
        text = embeddings.serialize_embedding(vector)
        self.assertEqual(embeddings.deserialize_embedding(text), vector)

    def test_deserialize_empty_returns_none(self):
        self.assertIsNone(embeddings.deserialize_embedding(""))


class SaveEmbeddingTest(unittest.TestCase):
    def test_zero_norm_vector_is_not_indexed(self):
        # コサイン距離が定義できないため索引には登録しない（records列には保存する）
        conn = make_conn()
        rec_id = insert_record(conn)

        embeddings.save_embedding(conn, rec_id, [0.0, 0.0, 0.0, 0.0])

        row = conn.execute(
            "SELECT embedding FROM records WHERE id = ?", (rec_id,)
        ).fetchone()
        self.assertEqual(
            embeddings.deserialize_embedding(row["embedding"]), [0.0, 0.0, 0.0, 0.0]
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_stores_embedding_column_and_vec_index(self):
        conn = make_conn()
        rec_id = insert_record(conn)

        embeddings.save_embedding(conn, rec_id, [1.0, 0.0, 0.0, 0.0])

        row = conn.execute(
            "SELECT embedding FROM records WHERE id = ?", (rec_id,)
        ).fetchone()
        self.assertEqual(
            embeddings.deserialize_embedding(row["embedding"]),
            [1.0, 0.0, 0.0, 0.0],
        )
        indexed = conn.execute(
            "SELECT rowid FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()
        self.assertIsNotNone(indexed)

    def test_replaces_existing_index_entry(self):
        conn = make_conn()
        rec_id = insert_record(conn)

        embeddings.save_embedding(conn, rec_id, [1.0, 0.0, 0.0, 0.0])
        embeddings.save_embedding(conn, rec_id, [0.0, 1.0, 0.0, 0.0])

        count = conn.execute(
            "SELECT COUNT(*) FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()[0]
        self.assertEqual(count, 1)
        row = conn.execute(
            "SELECT embedding FROM records WHERE id = ?", (rec_id,)
        ).fetchone()
        self.assertEqual(
            embeddings.deserialize_embedding(row["embedding"]),
            [0.0, 1.0, 0.0, 0.0],
        )


class ClearVecIndexTest(unittest.TestCase):
    def test_removes_from_index(self):
        conn = make_conn()
        rec_id = insert_record(conn)
        embeddings.save_embedding(conn, rec_id, [1.0, 0.0, 0.0, 0.0])

        embeddings.clear_vec_index(conn, rec_id)

        count = conn.execute(
            "SELECT COUNT(*) FROM vec_records WHERE rowid = ?", (rec_id,)
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_removing_absent_entry_is_a_noop(self):
        conn = make_conn()
        rec_id = insert_record(conn)
        # 索引に登録していない状態で呼んでもエラーにならない
        embeddings.clear_vec_index(conn, rec_id)


class TopKSimilarTest(unittest.TestCase):
    def _seed(self, conn, entries):
        """entries: [(instruction, output, vector), ...] をDBへ挿入する."""
        for instruction, output, vector in entries:
            rec_id = insert_record(conn, instruction, output)
            embeddings.save_embedding(conn, rec_id, vector)

    def test_ranks_by_similarity_descending(self):
        conn = make_conn()
        self._seed(
            conn,
            [
                ("遠い", "答え1", [0.0, 1.0, 0.0, 0.0]),
                ("近い", "答え2", [1.0, 0.0, 0.0, 0.0]),
                ("中間", "答え3", [1.0, 1.0, 0.0, 0.0]),
            ],
        )
        result = embeddings.top_k_similar(conn, [1.0, 0.0, 0.0, 0.0], k=3)
        instructions = [row["instruction"] for _, row in result]
        self.assertEqual(instructions, ["近い", "中間", "遠い"])

    def test_limits_to_k(self):
        conn = make_conn()
        self._seed(
            conn,
            [
                ("a", "答えa", [1.0, 0.0, 0.0, 0.0]),
                ("b", "答えb", [0.9, 0.1, 0.0, 0.0]),
                ("c", "答えc", [0.0, 1.0, 0.0, 0.0]),
            ],
        )
        result = embeddings.top_k_similar(conn, [1.0, 0.0, 0.0, 0.0], k=2)
        self.assertEqual(len(result), 2)

    def test_unvectorized_records_are_excluded(self):
        conn = make_conn()
        insert_record(conn, "未処理", "答え")  # ベクトル未設定
        self._seed(conn, [("処理済み", "答え", [1.0, 0.0, 0.0, 0.0])])

        result = embeddings.top_k_similar(conn, [1.0, 0.0, 0.0, 0.0], k=5)
        instructions = [row["instruction"] for _, row in result]
        self.assertEqual(instructions, ["処理済み"])

    def test_empty_index_returns_empty(self):
        conn = make_conn()
        self.assertEqual(
            embeddings.top_k_similar(conn, [1.0, 0.0, 0.0, 0.0], k=5), []
        )


class MaxSimilarityTest(unittest.TestCase):
    def test_returns_highest_score(self):
        conn = make_conn()
        for vector in (
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ):
            rec_id = insert_record(conn)
            embeddings.save_embedding(conn, rec_id, vector)

        score = embeddings.max_similarity(conn, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(score, 1.0)

    def test_empty_index_returns_zero(self):
        conn = make_conn()
        self.assertEqual(
            embeddings.max_similarity(conn, [1.0, 0.0, 0.0, 0.0]), 0.0
        )

    def test_zero_norm_indexed_vector_is_ignored(self):
        # ノルム0のベクトルとの類似度は計算不能（NULL）になるため0.0として扱う
        conn = make_conn()
        rec_id = insert_record(conn)
        embeddings.save_embedding(conn, rec_id, [0.0, 0.0, 0.0, 0.0])

        score = embeddings.max_similarity(conn, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(score, 0.0)

    def test_zero_norm_vector_does_not_shadow_a_real_match(self):
        # ノルム0のベクトルが索引に混ざっていても、実際に近い候補が
        # KNN検索で押しのけられずに見つかること（索引登録時に除外している）
        conn = make_conn()
        zero_id = insert_record(conn, "ゼロベクトル")
        embeddings.save_embedding(conn, zero_id, [0.0, 0.0, 0.0, 0.0])
        close_id = insert_record(conn, "近いベクトル")
        embeddings.save_embedding(conn, close_id, [1.0, 0.0, 0.0, 0.0])

        score = embeddings.max_similarity(conn, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
