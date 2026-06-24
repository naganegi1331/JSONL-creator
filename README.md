# LLM学習データ蓄積GUIツール

業務で得たナレッジやトラブルシューティングの記録をGUIから入力し、
LLMファインチューニング用のデータ（Alpaca形式）として蓄積するツールです。
データは **SQLite** で管理し、必要なときに **JSONL** へエクスポートできます。

## 特徴

- Tkinter / sqlite3（いずれもPython標準ライブラリ）製で外部依存なし
- Instruction / Input / Output の3欄を入力して1レコードずつ保存
- 保存済みデータの **一覧表示・編集・削除** に対応
- ボタン一つでDBの全件を `training_data.jsonl`（UTF-8 / `ensure_ascii=False`）へエクスポート
- exe化（PyInstaller等）してもスクリプト実行でも同じディレクトリにDB・出力を作成

## 動作環境

- 対象OS: Windows 10 / 11
- Python 3.x（Tkinter同梱の標準的なインストールでそのまま動作）

## 使い方

```sh
python jsonl_creator.py
```

左ペインに保存済みデータの一覧、右ペインに入力フォームが表示されます。

1. **Instruction**（必須）: 解決すべきメインの課題や質問文
2. **Input**（任意）: 前提条件、環境、エラーコード、ログなどの補足情報
3. **Output**（必須）: 最終的な回答、解決手順、社内仕様などの正解データ

### ボタン

- **保存**: 新規入力中はDBへ追加、一覧から行を選択して編集中はその行を更新します。
  InstructionまたはOutputが空欄（空白のみを含む）の場合は警告を表示して中断します。
- **新規入力**: フォームをクリアして新規入力モードに戻します。
- **選択行を削除**: 一覧で選択中のレコードを確認のうえ削除します。
- **JSONLにエクスポート**: DB内の全レコードを `training_data.jsonl` に書き出します。

一覧の行をクリックすると内容がフォームに読み込まれ、編集できます。

## データ管理

- 保存先DB: 実行ファイル/スクリプトと同じディレクトリの `training_data.db`
- 初回起動時、同ディレクトリに既存の `training_data.jsonl` があれば自動でDBへ取り込みます
  （DB新規作成時のみ。以降は取り込みません）

## 出力形式

エクスポートすると、1レコードにつき以下のJSONオブジェクトを1行ずつ書き出します。

```json
{"instruction": "入力値", "input": "入力値(空の場合は空文字)", "output": "入力値"}
```

## exe化

```sh
pip install pyinstaller
pyinstaller --onefile --noconsole jsonl_creator.py
```

生成された `dist/jsonl_creator.exe` を実行すると、exeと同じディレクトリに
`training_data.db` が作成され、エクスポート時に `training_data.jsonl` が出力されます。
