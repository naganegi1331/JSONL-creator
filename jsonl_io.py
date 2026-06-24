# -*- coding: utf-8 -*-
"""JSONLファイルのDBへの取り込み."""

import json
import os

from config import NEAR_DUPLICATE_THRESHOLD, get_export_path
from db import load_existing_keys, now_iso
from embeddings import (
    cosine_similarity,
    embedding_text,
    load_existing_embeddings,
    ollama_embed,
    serialize_embedding,
)


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
