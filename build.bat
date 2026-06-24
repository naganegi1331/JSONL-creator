@echo off
setlocal

rem ============================================================
rem  build.bat
rem  jsonl_creator.py を単一の exe にビルドする (Windows用)
rem  生成物: dist\jsonl_creator.exe
rem ============================================================

cd /d "%~dp0"

echo [1/3] Pythonを確認しています...
where python >nul 2>nul
if errorlevel 1 (
    echo エラー: Pythonが見つかりません。Pythonをインストールし、PATHに追加してください。
    exit /b 1
)

echo [2/3] PyInstallerを確認・インストールしています...
python -m pip show pyinstaller >nul 2>nul
if errorlevel 1 (
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo エラー: PyInstallerのインストールに失敗しました。
        exit /b 1
    )
)

echo [3/3] exeをビルドしています...
python -m PyInstaller --onefile --noconsole --name jsonl_creator jsonl_creator.py
if errorlevel 1 (
    echo エラー: ビルドに失敗しました。
    exit /b 1
)

echo.
echo ビルドが完了しました: dist\jsonl_creator.exe
endlocal
