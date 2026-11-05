import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_modes import MODES, validate_mode, retrieval_query
from qa_answers import build_qa_messages


class RequestModeTests(unittest.TestCase):
    def test_only_three_required_modes(self):
        self.assertEqual(set(MODES), {'route', 'recommendations', 'reference'})

    def test_unknown_and_retired_modes_rejected(self):
        for mode in ('transport', 'food', 'accommodation', 'tips', 'nearby', ''):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    validate_mode(mode)

    def test_reference_query_is_unchanged(self):
        self.assertEqual(retrieval_query('Что такое Пантеон?', 'reference'), 'Что такое Пантеон?')

    def test_recommendation_query_expanded(self):
        self.assertIn('attractions', retrieval_query('Что посмотреть?', 'recommendations'))

    def test_route_query_expanded(self):
        self.assertIn('itinerary', retrieval_query('На два дня', 'route'))

    def test_three_modes_create_prompts(self):
        p = SimpleNamespace(title='Paris', text='The Eiffel Tower is a landmark in Paris.')
        for mode in MODES:
            with self.subTest(mode=mode):
                messages = build_qa_messages('Расскажи о городе', [p], [], mode=mode)
                self.assertIn('Paris', messages[1]['content'])
                self.assertIn('Не рассчитывай расстояния', messages[0]['content'])

    def test_html_select_exposes_only_three_modes(self):
        import re
        html = (Path(__file__).resolve().parents[1] / 'static' / 'index.html').read_text(encoding='utf-8')
        select = html.split('<select id="request-mode"', 1)[1].split('</select>', 1)[0]
        self.assertEqual(set(re.findall(r'<option value="([^"]+)"', select)), set(MODES))

    def test_js_hints_expose_only_three_modes(self):
        script = (Path(__file__).resolve().parents[1] / 'static' / 'app.js').read_text(encoding='utf-8')
        fragment = script.split('const modeExamples = {', 1)[1].split('};', 1)[0]
        import re
        self.assertEqual(set(re.findall(r'^\s+(\w+):\[', fragment, flags=re.M)), set(MODES))


if __name__ == '__main__':
    unittest.main()
