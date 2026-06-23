# LLM学習データ蓄積GUIツール

サポート業務で得たナレッジやトラブルシューティングの記録をGUIから入力し、
LLMファインチューニング用のデータ（JSONL形式・Alpaca形式）として
自動整形・蓄積するツールです。

## 特徴

- Tkinter（Python標準ライブラリ）製で外部依存なし
- Instruction / Input / Output の3欄を入力して1レコードずつ追記
- `training_data.jsonl` にUTF-8（`ensure_ascii=False`）で追記保存
- exe化（PyInstaller等）してもスクリプト実行でも同じディレクトリに保存

## 動作環境

- 対象OS: Windows 10 / 11
- Python 3.x（Tkinter同梱の標準的なインストールでそのまま動作）

## 使い方

```sh
python jsonl_creator.py
```

1. **Instruction**（必須）: 解決すべきメインの課題や質問文
2. **Input**（任意）: 前提条件、環境、エラーコード、ログなどの補足情報
3. **Output**（必須）: 最終的な回答、解決手順、社内仕様などの正解データ

「JSONLに保存してクリア」ボタンで `training_data.jsonl` に1行追記し、
成功すると入力欄をリセットします。InstructionまたはOutputが
空欄（空白のみを含む）の場合は警告を表示して保存を中断します。

## 出力形式

実行ファイル/スクリプトと同じディレクトリの `training_data.jsonl` に、
1保存につき以下のJSONオブジェクトを1行ずつ追記します。

```json
{"instruction": "入力値", "input": "入力値(空の場合は空文字)", "output": "入力値"}
```

## exe化

```sh
pip install pyinstaller
pyinstaller --onefile --noconsole jsonl_creator.py
```

生成された `dist/jsonl_creator.exe` を実行すると、exeと同じ
ディレクトリに `training_data.jsonl` が作成・追記されます。
