import json
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import normalize_text, split_into_chunks, prepare


class TestPreparation(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  A\n  B\t C&nbsp;"), "A B C")

    def test_chunking_preserves_every_word(self):
        tokens = [f"word{i}" for i in range(24)]
        chunks = split_into_chunks(" ".join(tokens), words_per_chunk=10, overlap_words=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-2:], chunks[1].split()[:2])
        self.assertEqual(chunks[1].split()[-2:], chunks[2].split()[:2])
        rebuilt = chunks[0].split() + chunks[1].split()[2:] + chunks[2].split()[2:]
        self.assertEqual(rebuilt, tokens)

    def test_bad_overlap_raises(self):
        with self.assertRaises(ValueError):
            split_into_chunks("one two", 10, 10)

    def test_output_and_skipped_records(self):
        rows = [
            {"id": "42", "title": "City", "url": "https://example.org/city",
             "text": "A long description about walking routes and parks and museums " * 5,
             "lat": 51.2, "lon": 13.1},
            {"id": "42", "title": "City", "url": "https://example.org/city",
             "text": "A long description about walking routes and parks and museums " * 5,
             "lat": 51.2, "lon": 13.1},
            {"id": "43", "title": "Bad", "url": "https://example.org/bad",
             "text": "Short", "lat": 1, "lon": 1},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            stats = prepare(rows, path, words_per_chunk=15, overlap_words=3)
            self.assertEqual(stats["rows_seen"], 3)
            self.assertEqual(stats["articles_saved"], 1)
            self.assertEqual(stats["skipped_duplicate_id"], 1)
            self.assertEqual(stats["skipped_short_text"], 1)
            self.assertGreater(stats["chunks_saved"], 1)
            articles = [json.loads(x) for x in (path / "articles.jsonl").read_text().splitlines()]
            self.assertEqual(articles[0]["lat"], 51.2)
            self.assertEqual(articles[0]["license"], "CC BY-SA 4.0")
            chunks = [json.loads(x) for x in (path / "chunks.jsonl").read_text().splitlines()]
            self.assertTrue(chunks[0]["content"].startswith("Destination: City\n"))


if __name__ == "__main__":
    unittest.main()
