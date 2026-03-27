from __future__ import annotations

import unittest

from utils.video import _parse_ffprobe_csv_float


class VideoTests(unittest.TestCase):
    def test_parse_ffprobe_csv_float_ignores_trailing_separator(self) -> None:
        self.assertEqual(_parse_ffprobe_csv_float("0.000000,"), 0.0)

    def test_parse_ffprobe_csv_float_parses_plain_value(self) -> None:
        self.assertEqual(_parse_ffprobe_csv_float("1.040000"), 1.04)


if __name__ == "__main__":
    unittest.main()
