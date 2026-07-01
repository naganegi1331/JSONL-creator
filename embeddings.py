# -*- coding: utf-8 -*-
"""Ollamaによる埋め込みベクトルの生成・保存形式・sqlite-vecによる類似度計算."""

import json
import urllib.error
import urllib.request

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


def save_embedding(conn, record_id, vector):
    """埋め込みベクトルをrecords列とsqlite-vec索引（vec_records）の両方に保存する.

    既に索引エントリがあれば置き換える（編集後の再ベクトル化にも対応）。
    """
    conn.execute(
        "UPDATE records SET embedding = ? WHERE id = ?",
        (serialize_embedding(vector), record_id),
    )
    index_vector(conn, record_id, vector)


def index_vector(conn, record_id, vector):
    """ベクトルをsqlite-vec索引（vec_records）へ登録する（既存分は置き換え）.

    ノルム0のベクトルはコサイン距離が定義できずNULLになり、KNN検索で
    本来最も近いはずの候補より優先されてしまう（sqlite-vecの挙動）ため、
    索引には登録しない（コサイン類似度は常に0なので検索結果からの除外は
    実害がない）。
    """
    clear_vec_index(conn, record_id)
    if any(component != 0 for component in vector):
        conn.execute(
            "INSERT INTO vec_records(rowid, embedding) VALUES (?, ?)",
            (record_id, serialize_embedding(vector)),
        )


def clear_vec_index(conn, record_id):
    """sqlite-vec索引（vec_records）から該当レコードのベクトルを削除する.

    records.embedding列自体のクリアは呼び出し元の責務（本関数は索引のみ操作）。
    """
    conn.execute("DELETE FROM vec_records WHERE rowid = ?", (record_id,))


def top_k_similar(conn, query_vector, k):
    """クエリに近い順に上位k件を返す.

    sqlite-vecの索引テーブル（vec_records）に対するKNN検索で計算する
    （未ベクトル化のレコードは対象外）。戻り値は (score, row) を類似度の
    降順に最大k件並べたリスト。rowはid・instruction・outputを含むRow。
    """
    rows = conn.execute(
        "SELECT r.id, r.instruction, r.output, v.distance "
        "FROM vec_records v JOIN records r ON r.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance",
        (serialize_embedding(query_vector), k),
    ).fetchall()
    return [(1.0 - row["distance"], row) for row in rows]


def max_similarity(conn, query_vector):
    """クエリベクトルと索引済みベクトルとの最大コサイン類似度を返す（無ければ0.0）."""
    row = conn.execute(
        "SELECT distance FROM vec_records WHERE embedding MATCH ? AND k = 1 "
        "ORDER BY distance",
        (serialize_embedding(query_vector),),
    ).fetchone()
    if row is None or row["distance"] is None:
        return 0.0
    return 1.0 - row["distance"]
