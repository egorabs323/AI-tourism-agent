import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import web_app
from rag_pipeline import Passage
from reference_rag import ReferenceQuery
from recommendation_rag import RecommendationRequest


class RouterTests(unittest.TestCase):
    def run_reply(self, mode, documents):
        passage = Passage('Alpha Palace is a baroque palace built in 1700.', 'City',
                          'https://en.wikivoyage.org/wiki/City', 0, 'article:1')
        assistant = SimpleNamespace(titles={'City'}, articles=SimpleNamespace(city=Mock(return_value=(passage,))),
                                    retrieve=Mock(return_value=([passage], 'City')))
        with TemporaryDirectory() as folder, \
             patch.object(web_app, 'HISTORY_DB', Path(folder) / 'history.sqlite3'), \
             patch.object(web_app, 'get_assistant', return_value=assistant), \
             patch.object(web_app, 'search_documents', return_value=documents), \
             patch.object(web_app, 'reference_search_query', return_value=ReferenceQuery('City')), \
             patch.object(web_app, 'answer_route', return_value='route') as route, \
             patch.object(web_app, 'enrich_route_answer', return_value=('route', [passage])) as enrich, \
             patch.object(web_app, 'answer_factual', return_value='reference') as factual:
            web_app.generate_reply('test', 'City', 'City', 6, mode)
        return assistant, route, enrich, factual

    def test_route_reads_city_articles_without_vector_queries(self):
        assistant, route, enrich, factual = self.run_reply('route', [])
        assistant.retrieve.assert_not_called()
        route.assert_called_once()
        enrich.assert_called_once()
        factual.assert_not_called()

    def test_reference_mode_does_not_create_route(self):
        assistant, route, enrich, factual = self.run_reply('reference', [])
        assistant.retrieve.assert_called_once()
        route.assert_not_called()
        factual.assert_called_once()

    def test_uploaded_documents_still_use_document_answer(self):
        assistant, route, enrich, factual = self.run_reply('route', [
            Passage('User document text', 'Guide.txt', 'http://127.0.0.1/document', 0, 'doc_1')])
        assistant.articles.city.assert_not_called()
        assistant.retrieve.assert_not_called()
        route.assert_not_called()
        factual.assert_called_once()

    def test_recommendations_use_their_query_and_verified_answer_handler(self):
        passage = Passage('Alpha Museum is a museum of modern art.', 'City',
                          'https://example.test/city', 0, 'article:1')
        request = RecommendationRequest('City art museums', count=2)
        assistant = SimpleNamespace(titles={'City'}, encoder=Mock(),
                                    articles=SimpleNamespace(city=Mock(return_value=(passage,))),
                                    retrieve=Mock(return_value=([passage], 'City')))
        with TemporaryDirectory() as folder, \
             patch.object(web_app, 'HISTORY_DB', Path(folder) / 'history.sqlite3'), \
             patch.object(web_app, 'get_assistant', return_value=assistant), \
             patch.object(web_app, 'search_documents', return_value=[]), \
             patch.object(web_app, 'prepare_recommendations', return_value=request), \
             patch.object(web_app, 'answer_recommendations', return_value=('verified', [])) as recommend, \
             patch.object(web_app, 'answer_route') as route, \
             patch.object(web_app, 'answer_factual') as factual:
            answer, _, _ = web_app.generate_reply('test', 'Два музея', 'City', 6, 'recommendations')
        self.assertEqual(answer, 'verified')
        assistant.retrieve.assert_called_once_with('City art museums', destination='City', top_k=6,
                                                   mode='recommendations')
        recommend.assert_called_once()
        self.assertIs(recommend.call_args.kwargs['request'], request)
        route.assert_not_called()
        factual.assert_not_called()
