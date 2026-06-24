# -*- coding: utf-8 -*-
"""LLM学習データ蓄積GUIツール（エントリポイント）.

サポート業務で得たナレッジやトラブルシューティングの記録を、エンジニアが
直感的なGUIから入力し、LLMファインチューニング用のデータ（Alpaca形式）
として蓄積する。データはSQLiteで管理し、JSONL形式でエクスポートできる。

対象OS : Windows 10 / 11
GUI    : Tkinter（Python標準ライブラリ）
DB     : SQLite（sqlite3・標準ライブラリ）
"""

import os
import tkinter as tk
from tkinter import messagebox

from config import EXPORT_FILENAME, get_db_path
from db import connect_db, init_db
from gui import JsonlCreatorApp
from jsonl_io import import_legacy_jsonl


def main():
    db_existed = os.path.exists(get_db_path())
    conn = connect_db()
    init_db(conn)

    imported = 0
    if not db_existed:
        # DB新規作成時のみ、既存の training_data.jsonl を取り込む
        imported = import_legacy_jsonl(conn)

    root = tk.Tk()
    JsonlCreatorApp(root, conn)
    if imported:
        messagebox.showinfo(
            "インポート",
            f"既存の {EXPORT_FILENAME} から {imported} 件を取り込みました。",
        )
    try:
        root.mainloop()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
