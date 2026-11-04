"""Unit tests; run after dependencies of the main project have been installed."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from rag_pipeline import Passage
from structured_routes import Attraction, grounded_description, extract_route_candidates, answer_route


class GroundedDescriptionsTests(unittest.TestCase):
    def test_no_false_aerodromes(self):
        a = Attraction('Parco degli Acquedotti', 1, 'Rome/South', 'https://example.com',
                       'Parco degli Acquedotti Park contains preserved aqueducts in Rome.')
        d = grounded_description(a)
        self.assertIn('акведук', d)
        self.assertNotIn('аэродром', d)

    def test_no_unverified_details(self):
        a = Attraction('Pantheon', 1, 'Rome', 'https://example.com', 'Pantheon can be found downtown.')
        self.assertEqual(grounded_description(a), '')

    def test_ignores_hallucinated_llm_description(self):
        passages = [Passage('Colosseum is a landmark of ancient Rome. ' * 3,
                            'Rome', 'https://example.com', 0.1, 'abc')]
        response = {'attractions': [{'name': 'Colosseum', 'source': 1,
                                     'description': 'В Колизее проходили футбольные матчи в 2025 году.'}]}
        with TemporaryDirectory() as folder:
            with patch('structured_routes._ollama_json', return_value=response), \
                 patch('structured_routes.ollama_configuration', return_value=('http://localhost:11434', 'qwen3:1.7b')):
                attractions, descriptions = extract_route_candidates(passages, 'Rome', 1, None,
                                                                       cache_dir=Path(folder))
        self.assertEqual([a.name for a in attractions], ['Colosseum'])
        self.assertNotIn('футболь', descriptions['colosseum'])

    def test_sources_and_names_from_source_only(self):
        passages = [Passage('Pantheon is an ancient Roman temple.', 'Rome',
                            'https://example.org/rome', .1, 'id')]
        response = {'attractions': [{'name': 'Pantheon', 'source': 1, 'description': 'Летающий дворец'},
                                    {'name': 'Fake Vatican Castle', 'source': 1}]}
        with patch('structured_routes.extract_route_candidates', return_value=(
                [Attraction('Pantheon', 1, 'Rome', 'https://example.org/rome',
                            'Pantheon is an ancient Roman temple.')],
                {'pantheon': 'Историческое сооружение Рима.'})):
            output = answer_route('маршрут на два дня', passages, 'Rome')
        self.assertIn('Pantheon', output)
        self.assertNotIn('Fake Vatican Castle', output)
        self.assertIn('https://example.org/rome', output)


if __name__ == '__main__':
    unittest.main()
