import io
import json
import unittest
from unittest.mock import patch
from rag_pipeline import Passage, answer_with_llm, ollama_configuration


class FakeResponse:
    def __enter__(self):
        return io.BytesIO(b'{"message": {"content": "Test answer"}}')
    def __exit__(self, *args):
        return False


class TestOllama(unittest.TestCase):
    def test_default_config(self):
        with patch.dict('os.environ', {'OLLAMA_URL': 'http://127.0.0.1:11434', 'OLLAMA_MODEL': 'qwen3:1.7b'}):
            self.assertEqual(ollama_configuration(), ('http://127.0.0.1:11434', 'qwen3:1.7b'))

    def test_nonlocal_url_is_rejected(self):
        with patch.dict('os.environ', {'OLLAMA_URL': 'https://example.org'}):
            with self.assertRaises(ValueError):
                ollama_configuration()

    @patch('urllib.request.urlopen', return_value=FakeResponse())
    def test_local_call_rag(self, mock_open):
        passage = Passage('Colosseum description', 'Rome', 'https://example.org/Rome', 0.1, 'chunk1')
        reply = answer_with_llm('Что посмотреть?', [passage], [], model='qwen3:1.7b')
        self.assertEqual(reply, 'Test answer')
        request = mock_open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        body = json.loads(request.data)
        self.assertFalse(body['think'])
        self.assertEqual(body['model'], 'qwen3:1.7b')
        self.assertIn('Colosseum description', body['messages'][-1]['content'])

    @patch('urllib.request.urlopen', return_value=FakeResponse())
    def test_without_rag_has_no_source(self, mock_open):
        answer_with_llm('Привет', [], [], without_rag=True)
        body = json.loads(mock_open.call_args.args[0].data)
        self.assertNotIn('ИСТОЧНИКИ:', body['messages'][-1]['content'])

    def test_empty_passages_does_not_call_ollama(self):
        result = answer_with_llm('Что посмотреть?', [], [], without_rag=False)
        self.assertIn('не найдены', result)

if __name__ == '__main__':
    unittest.main()
