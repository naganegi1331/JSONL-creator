# -*- coding: utf-8 -*-
"""Ollamaによる埋め込みベクトルの生成・保存形式・類似度計算."""

import json
import urllib.error
import urllib.request

import numpy as np

from config import OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_TIMEOUT


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


def _cosine_batch(query_vector, vectors):
    """クエリベクトルと複数ベクトルのコサイン類似度を行列演算でまとめて返す.

    戻り値は list[float]。ノルム0のベクトルとの類似度は0.0として扱う。
    """
    if not vectors:
        return []
    query = np.asarray(query_vector, dtype=float)
    matrix = np.asarray(vectors, dtype=float)
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    dots = matrix @ query
    scores = np.zeros_like(dots)
    nonzero = denom != 0
    scores[nonzero] = dots[nonzero] / denom[nonzero]
    return scores.tolist()


def top_k_similar(query_vector, candidates, k):
    """クエリに近い順に上位k件を返す.

    candidates は (payload, vector) のリスト。戻り値は (score, payload) を
    類似度の降順に最大k件並べたリスト。payload同士は比較しない（スコアの
    みでソートする）ので、payloadがsqlite3.Row等でも安全。
    """
    if not candidates:
        return []
    payloads = [payload for payload, _ in candidates]
    vectors = [vector for _, vector in candidates]
    scores = _cosine_batch(query_vector, vectors)
    ranked = sorted(
        zip(scores, payloads), key=lambda pair: pair[0], reverse=True
    )
    return ranked[:k]


def max_similarity(query_vector, vectors):
    """クエリベクトルと各ベクトルの最大コサイン類似度を返す（無ければ0.0）."""
    scores = _cosine_batch(query_vector, vectors)
    return max(scores) if scores else 0.0


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
