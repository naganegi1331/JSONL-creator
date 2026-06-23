# -*- coding: utf-8 -*-
"""LLM学習データ蓄積GUIツール.

サポート業務で得たナレッジやトラブルシューティングの記録を、エンジニアが
直感的なGUIから入力し、LLMファインチューニング用のデータ（JSONL形式・
Alpaca形式）として自動整形・蓄積する。

対象OS : Windows 10 / 11
GUI    : Tkinter（Python標準ライブラリ）
"""

import json
import os
import sys
import tkinter as tk
from tkinter import messagebox


OUTPUT_FILENAME = "training_data.jsonl"


def get_base_dir():
    """出力ファイルの基準ディレクトリを返す.

    exe実行時（PyInstaller等でfrozen）とスクリプト実行時の両方で、
    実行ファイル/スクリプトと同じディレクトリを正しく解決する。
    """
    if getattr(sys, "frozen", False):
        # exe化されている場合は実行ファイルのあるディレクトリ
        return os.path.dirname(sys.executable)
    # 通常のスクリプト実行時はこのファイルのあるディレクトリ
    return os.path.dirname(os.path.abspath(__file__))


def get_output_path():
    """training_data.jsonl のフルパスを返す."""
    return os.path.join(get_base_dir(), OUTPUT_FILENAME)


class JsonlCreatorApp:
    """入力フォームと保存処理を提供するメインアプリケーション."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM学習データ蓄積ツール")
        self.root.geometry("700x600")
        self.root.minsize(500, 400)

        self._build_widgets()

    def _build_widgets(self):
        """ウィジェットを上から順に配置する.

        gridのrow/columnにweightを設定し、各テキストエリアが
        ウィンドウのリサイズに追従して伸縮するようにする。
        """
        self.root.columnconfigure(0, weight=1)

        # Instruction（必須）
        self.root.rowconfigure(1, weight=1)
        tk.Label(
            self.root,
            text="Instruction (Q: 顧客からの質問や発生した課題)",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.instruction_text = tk.Text(self.root, height=6, wrap="word")
        self.instruction_text.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=(0, 5)
        )

        # Input（任意）
        self.root.rowconfigure(3, weight=1)
        tk.Label(
            self.root,
            text="Input (Context: 前提条件、環境、エラーコードなど)",
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 0))
        self.input_text = tk.Text(self.root, height=6, wrap="word")
        self.input_text.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 5))

        # Output（必須）
        self.root.rowconfigure(5, weight=1)
        tk.Label(
            self.root,
            text="Output (A: 解決策、実際の対応内容)",
            anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 0))
        self.output_text = tk.Text(self.root, height=6, wrap="word")
        self.output_text.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 5))

        # 保存ボタン
        self.save_button = tk.Button(
            self.root,
            text="JSONLに保存してクリア",
            command=self.save_entry,
        )
        self.save_button.grid(row=6, column=0, sticky="ew", padx=10, pady=10)

    def _get_value(self, widget):
        """テキストエリアの値を取得し、末尾の改行を除去して返す."""
        # Textウィジェットは末尾に自動的に改行を持つため rstrip("\n") で除去する
        return widget.get("1.0", "end").rstrip("\n")

    def save_entry(self):
        """入力内容をバリデーションし、JSONL形式でファイルへ追記する."""
        instruction = self._get_value(self.instruction_text)
        input_value = self._get_value(self.input_text)
        output = self._get_value(self.output_text)

        # 必須入力チェック（空白文字のみも空欄とみなす）
        if not instruction.strip() or not output.strip():
            messagebox.showwarning(
                "入力エラー", "必須項目が入力されていません"
            )
            return

        record = {
            "instruction": instruction,
            "input": input_value,
            "output": output,
        }

        # json.dumps で特殊文字を正しくエスケープし、日本語はエスケープしない
        line = json.dumps(record, ensure_ascii=False) + "\n"

        try:
            with open(get_output_path(), "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            # 書き込み権限なし、ネットワーク切断などで保存に失敗した場合。
            # 入力データは画面上に保持したまま処理を中断する。
            messagebox.showerror(
                "保存エラー",
                "ファイルの保存に失敗しました。\n"
                "書き込み権限やネットワーク接続を確認してください。\n\n"
                f"詳細: {e}",
            )
            return

        # 成功時：完了を通知し、すべての入力欄をクリアする
        messagebox.showinfo("保存完了", "データを保存しました。")
        self._clear_inputs()

    def _clear_inputs(self):
        """すべての入力欄をリセットする."""
        self.instruction_text.delete("1.0", "end")
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.instruction_text.focus_set()


def main():
    root = tk.Tk()
    JsonlCreatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
