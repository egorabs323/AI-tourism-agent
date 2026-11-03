import unittest
from rag_pipeline import detect_destination, destination_titles, resolve_title, build_context, Passage


class TestRagPipeline(unittest.TestCase):
    def setUp(self):
        self.titles = {"Rome", "Rome/Colosseo", "Rome/Old Rome", "Marche", "Paris", "Aachen"}

    def test_detect_russian_city(self):
        self.assertEqual(detect_destination("Какие достопримечательности посетить в Риме за два дня?", self.titles), "Rome")

    def test_detect_english_city(self):
        self.assertEqual(detect_destination("What to see in Aachen?", self.titles), "Aachen")

    def test_title_filter(self):
        self.assertEqual(destination_titles("Rome", self.titles), ["Rome", "Rome/Colosseo", "Rome/Old Rome"])
        self.assertNotIn("Marche", destination_titles("Rome", self.titles))

    def test_missing_city(self):
        self.assertIsNone(resolve_title("Atlantis", self.titles))
        with self.assertRaises(ValueError):
            destination_titles("Atlantis", self.titles)

    def test_context_sources(self):
        context = build_context([Passage("A museum", "Rome", "https://x.example", .1, "1")])
        self.assertIn("[Источник 1]", context)
        self.assertIn("https://x.example", context)


if __name__ == "__main__":
    unittest.main()
