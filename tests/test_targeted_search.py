"""Лёгкие тесты поиска без SentenceTransformers и ChromaDB."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

fake = types.ModuleType('build_vector_db')
fake.MODEL_NAME = 'unused'
fake.COLLECTION = 'unused'
fake.create_collection = lambda db: None
sys.modules.setdefault('build_vector_db', fake)
path = Path(__file__).resolve().parents[1] / 'rag_pipeline.py'
spec = importlib.util.spec_from_file_location('rag_pipeline', path)
module = importlib.util.module_from_spec(spec)
sys.modules['rag_pipeline'] = module
spec.loader.exec_module(module)

class Vectors:
    def __init__(self):
        self.calls = []
    def encode(self, texts, normalize_embeddings=False):
        self.calls.append(texts)
        return [type('V', (), {'tolist': lambda self: [0.1, 0.2]})() for _ in texts]

class Collection:
    def __init__(self):
        self.calls = []
    def count(self):
        return 100
    def query(self, query_embeddings, n_results, where, include):
        self.calls.append((query_embeddings, n_results, where))
        ids = ['a','b','c','d']
        docs = [
            'Day trip outside the city via airport bus lines.',
            'Colosseum is a famous monument and main landmark. Roman Forum and basilica.',
            'Museum with major historic architecture and a beautiful piazza.',
            'Train timetable tickets and discounts.',
        ]
        metas = [{'title':'Rome','url':'https://example.org/rome'},
                 {'title':'Rome/Colosseo','url':'https://example.org/colosseo'},
                 {'title':'Rome/Old Rome','url':'https://example.org/old'},
                 {'title':'Rome','url':'https://example.org/train'}]
        n = len(query_embeddings)
        return {'ids': [ids]*n, 'documents':[docs]*n,
                'metadatas':[metas]*n, 'distances':[[.12,.15,.17,.2]]*n}

class SearchTests(unittest.TestCase):
    def setUp(self):
        self.assistant=module.RagAssistant.__new__(module.RagAssistant)
        self.assistant.titles={'Rome','Rome/Colosseo','Rome/Old Rome'}
        self.assistant.encoder=Vectors()
        self.assistant.collection=Collection()
    def test_route_uses_three_short_queries_and_prefers_attractions(self):
        found, city = self.assistant.retrieve('Составь маршрут по Риму на два дня', 'Rome', 3)
        self.assertEqual(city,'Rome')
        self.assertEqual(len(self.assistant.encoder.calls[0]),3)
        self.assertEqual(len(self.assistant.collection.calls),1)
        self.assertIn('Colosseum',found[0].text)
        self.assertEqual(self.assistant.collection.calls[0][2]['title']['$in'],
                         ['Rome','Rome/Colosseo','Rome/Old Rome'])
    def test_regular_question_single_query(self):
        self.assistant.retrieve('What is the city history?', 'Rome', 3)
        self.assertEqual(len(self.assistant.encoder.calls[0]),1)
    def test_rejects_top_k_zero(self):
        with self.assertRaises(ValueError):
            self.assistant.retrieve('маршрут','Rome',0)

if __name__=='__main__':
    unittest.main()
