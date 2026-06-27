# -*- coding: utf-8 -*-
"""config.py のパス解決・定数のテスト."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class ConfigConstantsTest(unittest.TestCase):
    def test_filenames(self):
        self.assertEqual(config.DB_FILENAME, "training_data.db")
        self.assertEqual(config.EXPORT_FILENAME, "training_data.jsonl")

    def test_near_duplicate_threshold_is_valid_ratio(self):
        # コサイン類似度の閾値なので 0〜1 の範囲に収まっているはず
        self.assertGreater(config.NEAR_DUPLICATE_THRESHOLD, 0.0)
        self.assertLessEqual(config.NEAR_DUPLICATE_THRESHOLD, 1.0)

    def test_ollama_settings_present(self):
        self.assertTrue(config.OLLAMA_API_URL.startswith("http"))
        self.assertTrue(config.OLLAMA_EMBEDDING_MODEL)
        self.assertIsInstance(config.OLLAMA_TIMEOUT, int)


class ConfigPathTest(unittest.TestCase):
    def test_base_dir_exists(self):
        base = config.get_base_dir()
        self.assertTrue(os.path.isdir(base))

    def test_db_path_under_base_dir(self):
        path = config.get_db_path()
        self.assertEqual(os.path.dirname(path), config.get_base_dir())
        self.assertTrue(path.endswith(config.DB_FILENAME))

    def test_export_path_under_base_dir(self):
        path = config.get_export_path()
        self.assertEqual(os.path.dirname(path), config.get_base_dir())
        self.assertTrue(path.endswith(config.EXPORT_FILENAME))

    def test_frozen_mode_uses_executable_dir(self):
        # exe実行（PyInstaller等でfrozen）時は実行ファイルのディレクトリを使う
        original_frozen = getattr(sys, "frozen", None)
        original_executable = sys.executable
        try:
            sys.frozen = True
            sys.executable = os.path.join("某", "場所", "app.exe")
            self.assertEqual(config.get_base_dir(), os.path.join("某", "場所"))
        finally:
            sys.executable = original_executable
            if original_frozen is None:
                del sys.frozen
            else:
                sys.frozen = original_frozen


if __name__ == "__main__":
    unittest.main()
