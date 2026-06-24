# -*- coding: utf-8 -*-
"""パス解決と各種定数."""

import os
import sys


DB_FILENAME = "training_data.db"
EXPORT_FILENAME = "training_data.jsonl"

# Ollama（ローカルLLM）の埋め込みAPI設定。
# 事前に `ollama pull nomic-embed-text` 等でモデルを取得し、
# Ollamaをローカルで起動しておく必要がある。
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT = 10  # 秒

# インポート時、この類似度（コサイン類似度）以上を「類似重複」として報告する。
NEAR_DUPLICATE_THRESHOLD = 0.93


def get_base_dir():
    """基準ディレクトリ（DB・エクスポート先の親）を返す.

    exe実行時（PyInstaller等でfrozen）とスクリプト実行時の両方で、
    実行ファイル/スクリプトと同じディレクトリを正しく解決する。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_db_path():
    """SQLite DBファイルのフルパスを返す."""
    return os.path.join(get_base_dir(), DB_FILENAME)


def get_export_path():
    """エクスポート先 training_data.jsonl のフルパスを返す."""
    return os.path.join(get_base_dir(), EXPORT_FILENAME)
