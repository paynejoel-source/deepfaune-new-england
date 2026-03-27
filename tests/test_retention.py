from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from utils.retention import prune_old_files


class RetentionTests(unittest.TestCase):
    def test_prune_old_files_deletes_old_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_file = root / "old.json"
            new_file = root / "new.json"
            nested_old = root / "nested" / "older.json"
            nested_old.parent.mkdir(parents=True, exist_ok=True)

            old_file.write_text("old", encoding="utf-8")
            new_file.write_text("new", encoding="utf-8")
            nested_old.write_text("older", encoding="utf-8")

            old_age = time.time() - (31 * 86400)
            os.utime(old_file, (old_age, old_age))
            os.utime(nested_old, (old_age, old_age))

            deleted = prune_old_files(root, 30)

            self.assertEqual(set(deleted), {old_file, nested_old})
            self.assertFalse(old_file.exists())
            self.assertFalse(nested_old.exists())
            self.assertTrue(new_file.exists())
            self.assertFalse((root / "nested").exists())


if __name__ == "__main__":
    unittest.main()
