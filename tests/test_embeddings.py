# -*- coding: utf-8 -*-
"""embeddings.py のテキスト整形・直列化・類似度計算のテスト.

Ollamaへの実通信は行わず、純粋関数（NumPy計算）とDB読み込みのみを検証する。
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import embeddings


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


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


class CosineBatchTest(unittest.TestCase):
    def test_identical_vector_scores_one(self):
        scores = embeddings._cosine_batch([1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(scores[0], 1.0)

    def test_orthogonal_vector_scores_zero(self):
        scores = embeddings._cosine_batch([1.0, 0.0], [[0.0, 1.0]])
        self.assertAlmostEqual(scores[0], 0.0)

    def test_opposite_vector_scores_minus_one(self):
        scores = embeddings._cosine_batch([1.0, 0.0], [[-1.0, 0.0]])
        self.assertAlmostEqual(scores[0], -1.0)

    def test_zero_norm_vector_scores_zero(self):
        # ノルム0のベクトルとの類似度は0.0（ゼロ除算しない）
        scores = embeddings._cosine_batch([1.0, 0.0], [[0.0, 0.0]])
        self.assertAlmostEqual(scores[0], 0.0)

    def test_empty_candidates_returns_empty(self):
        self.assertEqual(embeddings._cosine_batch([1.0], []), [])


class TopKSimilarTest(unittest.TestCase):
    def test_ranks_by_similarity_descending(self):
        query = [1.0, 0.0]
        candidates = [
            ("遠い", [0.0, 1.0]),
            ("近い", [1.0, 0.0]),
            ("中間", [1.0, 1.0]),
        ]
        result = embeddings.top_k_similar(query, candidates, k=3)
        payloads = [payload for _, payload in result]
        self.assertEqual(payloads, ["近い", "中間", "遠い"])

    def test_limits_to_k(self):
        query = [1.0, 0.0]
        candidates = [("a", [1.0, 0.0]), ("b", [0.9, 0.1]), ("c", [0.0, 1.0])]
        result = embeddings.top_k_similar(query, candidates, k=2)
        self.assertEqual(len(result), 2)

    def test_empty_candidates(self):
        self.assertEqual(embeddings.top_k_similar([1.0], [], k=5), [])


class MaxSimilarityTest(unittest.TestCase):
    def test_returns_highest_score(self):
        score = embeddings.max_similarity(
            [1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        )
        self.assertAlmostEqual(score, 1.0)

    def test_empty_returns_zero(self):
        self.assertEqual(embeddings.max_similarity([1.0, 0.0], []), 0.0)


class LoadExistingEmbeddingsTest(unittest.TestCase):
    def test_loads_only_vectorized_rows(self):
        conn = make_conn()
        db.init_db(conn)
        ts = db.now_iso()
        conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ベクトルあり", "", "回答", "", embeddings.serialize_embedding([1.0, 2.0]), ts, ts),
        )
        conn.execute(
            "INSERT INTO records "
            "(instruction, input, output, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ベクトルなし", "", "回答", "", "", ts, ts),
        )
        conn.commit()

        result = embeddings.load_existing_embeddings(conn)
        self.assertEqual(len(result), 1)
        rec_id, vector = result[0]
        self.assertEqual(vector, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
