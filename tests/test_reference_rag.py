"""Synthetic reference RAG regressions: no live model or production city lists."""
import json
import unittest
from unittest.mock import Mock

from rag_pipeline import Passage
from reference_rag import (ReferenceQuery, reference_search_query, reference_context,
                           answer_reference, reference_sources, asks_current_values,
                           _valid_fact, _prepared_fact, NO_ANSWER, CURRENT_NOTICE)


def passage(text, title='City', url='https://example.test/city'):
    return Passage(text, title, url, 0.1, '1')


class ReferenceTests(unittest.TestCase):
    def test_rewrite_uses_only_user_history(self):
        generate = Mock(return_value={'entity': 'Alpha Tower', 'query': 'Alpha Tower height'})
        history = [{'role': 'user', 'content': 'Расскажи об Alpha Tower'},
                   {'role': 'assistant', 'content': 'Выдуманный факт про другой объект.'}]
        q = reference_search_query('Какова её высота?', history, 'City', generator=generate)
        self.assertIn('Alpha Tower', q.names)
        payload = json.loads(generate.call_args_list[0].args[1])
        self.assertEqual(payload['previous_user_questions'], ['Расскажи об Alpha Tower'])
        self.assertNotIn('Выдуманный', generate.call_args.args[1])

    def test_pronoun_followup_inherits_previous_english_name(self):
        generate = Mock(side_effect=[{'entity': 'Альфа', 'query': 'What is interesting about its dome?'},
                                     {'entity': 'Alpha Palace', 'query': 'What is Alpha Palace?'}])
        history = [{'role': 'user', 'content': 'Расскажи об Alpha Palace'},
                   {'role': 'assistant', 'content': 'Построен в выдуманном году.'}]
        q = reference_search_query('Чем интересен его купол?', history, 'City', generator=generate)
        self.assertIn('Alpha Palace', q.names)
        self.assertIn('Alpha Palace', q.query)

    def test_rewrite_cannot_add_a_year(self):
        generate = Mock(return_value={'entity': 'Alpha Gate', 'query': 'Alpha Gate built 1700'})
        q = reference_search_query('Когда построены Alpha Gate?', [], generator=generate)
        self.assertNotIn('1700', q.query)

    def test_english_query_repairs_untranslated_entity_field(self):
        generate = Mock(return_value={'entity': 'Альфа', 'query': 'What is the Alpha Museum in City?'})
        q = reference_search_query('Что такое Альфа?', [], 'City', generator=generate)
        self.assertIn('Alpha Museum', q.names)

    def test_nearby_mention_cannot_own_date(self):
        p = passage('See * Beta Palace was built in 1750 near Alpha Gate. '
                    'Alpha Gate * Alpha Gate.jpg The gate was designed in 1791 by Ada Stone.')
        result = reference_context('Когда построены Alpha Gate?', ReferenceQuery('Alpha Gate', ('Alpha Gate',)), [p])
        self.assertTrue(result)
        self.assertTrue(all('1750' not in r.text for r in result))
        self.assertIn('1791', result[0].text)

    def test_second_suggested_name_is_not_accepted_as_an_alias(self):
        p = passage('See * Alpha Gate.jpg The gate was designed in 1791. '
                    '* Beta Gate.jpg The gate was built in 1750.')
        result = reference_context('Когда построены Alpha Gate?', ReferenceQuery('gate', ('Alpha Gate', 'Beta Gate')), [p])
        self.assertTrue(result)
        self.assertTrue(all('1750' not in r.text for r in result))

    def test_city_prefix_and_plural_are_resolved_to_source_name(self):
        p = passage('See * Astronomical Clock.jpg The clock was built in 1410. '
                    'The dial shows the sun and moon.')
        query = ReferenceQuery('clock', ('City Astronomical Clocks',))
        result = reference_context('Что показывают часы?', query, [p])
        self.assertTrue(any('sun and moon' in r.text for r in result))

    def test_unknown_quoted_name_refuses_even_if_phone_found(self):
        p = passage('Embassies * Embassy Alpha has a telephone number +39 06 12345.')
        query = ReferenceQuery('phone', ('Embassy Alpha',))
        self.assertEqual(reference_context('Телефон бюро «Астра-917»?', query, [p]), [])
        self.assertEqual(answer_reference('Телефон бюро «Астра-917»?', []), NO_ANSWER)

    def test_russian_user_document_keeps_its_named_facts(self):
        p = Passage('Обсерватория «Лунный маяк» открыта в 1987 году. В коллекции 42 метеорита.',
                    'Документ: guide.txt', 'http://localhost/document', 0.1, 'doc_1')
        query = ReferenceQuery('Lunar Lighthouse', ('Лунный маяк', 'Lunar Lighthouse'))
        facts = reference_context('Когда открыта обсерватория «Лунный маяк»?', query, [p])
        self.assertTrue(any('1987' in f.text for f in facts))
        generate = Mock(return_value={'1': 'Обсерватория открыта в 1987 году.'})
        answer = answer_reference('Когда открыта обсерватория?', facts, generate)
        self.assertIn('1987', answer)
        self.assertIn('[Источник 1]', answer)

    def test_known_document_object_with_missing_property_refuses(self):
        p = passage('Обсерватория «Лунный маяк» открыта в 1987 году. В коллекции 42 метеорита.')
        query = ReferenceQuery('observatory architect', ('Лунный маяк',))
        self.assertEqual(reference_context('Кто архитектор обсерватории «Лунный маяк»?', query, [p]), [])

    def test_document_quantity_does_not_return_foundation_date(self):
        p = passage('Обсерватория «Лунный маяк» открыта в 1987 году. В коллекции 42 метеорита.')
        query = ReferenceQuery('meteorites', ('Лунный маяк',))
        facts = reference_context('Сколько там метеоритов?', query, [p])
        self.assertTrue(facts)
        self.assertTrue(all('42' in f.text for f in facts))

    def test_unmatched_source_name_cannot_be_invented_by_resolver(self):
        p = passage('Alpha Bridge connects the Old Town with New Town.')
        generate = Mock(return_value={'name': 'Invented Bridge'})
        query = ReferenceQuery('Unknown Bridge', ('Unknown Bridge',))
        self.assertEqual(reference_context('Расскажи об Unknown Bridge', query, [p], generator=generate), [])

    def test_resolver_menu_excludes_combined_landmark_captions(self):
        p = passage('See Alpha Gate The Beta Museum * Alpha Gate.jpg The gate was designed in 1791.')
        generate = Mock(return_value={'name': 'Alpha Gate'})
        query = ReferenceQuery('Unknown Gate', ('Unknown Gate',))
        result = reference_context('Когда построены ворота Альфа?', query, [p], generator=generate)
        self.assertTrue(result)
        candidates = json.loads(generate.call_args.args[1])['source_names']
        self.assertEqual(candidates, ['Alpha Gate'])

    def test_unique_identical_linking_clause_recovers_lost_listing_title(self):
        overview = passage('Alpha Bridge connects Old Town with New Town.')
        listing = passage('See * View from Gamma Tower.jpg connects the Old Town with New Town. '
                          'Constructed between 1357 and 1402, it is a historic stone bridge.',
                          'City/West', 'https://example.test/west')
        q = ReferenceQuery('Alpha Bridge', ('Alpha Bridge',))
        result = reference_context('Когда началось строительство Alpha Bridge?', q, [], [overview, listing])
        self.assertTrue(any('1357' in p.text for p in result))

    def test_ambiguous_linking_clause_cannot_transfer_date(self):
        overview = passage('Alpha Bridge connects Old Town with New Town. '
                           'Beta Bridge connects Old Town with New Town.')
        listing = passage('See * View.jpg connects Old Town with New Town. '
                          'Constructed between 1357 and 1402, it is a historic stone bridge.')
        q = ReferenceQuery('Alpha Bridge', ('Alpha Bridge',))
        result = reference_context('Когда построен Alpha Bridge?', q, [], [overview, listing])
        self.assertTrue(all('1357' not in p.text for p in result))

    def test_dome_facts_exclude_booking_instructions(self):
        p = passage('See * Alpha Palace.jpg This building houses a museum. '
                    'Alpha Palace has a glass dome designed by Ada Stone and completed in 1999. '
                    'You can book a visit to the dome in advance.')
        q = ReferenceQuery('Alpha Palace', ('Alpha Palace',))
        result = reference_context('Чем интересен его купол?', q, [p])
        self.assertTrue(any('1999' in p.text for p in result))
        self.assertTrue(all('book' not in p.text for p in result))

    def test_display_facts_exclude_unrequested_clock_show(self):
        p = passage('See * Astronomical Clock.jpg The dial shows the sun and moon. '
                    'There is a 12-month calendar. Every hour a rooster crows.')
        q = ReferenceQuery('clock', ('Astronomical Clock',))
        result = reference_context('Что показывают часы?', q, [p])
        self.assertTrue(result)
        self.assertTrue(all('rooster' not in p.text for p in result))

    def test_current_prices_are_not_confirmed_from_static_data(self):
        self.assertTrue(asks_current_values('Сколько стоит билет сегодня?'))
        generate = Mock()
        self.assertEqual(answer_reference('Сколько стоит билет сегодня?', [passage('Fee: €18.')], generate), CURRENT_NOTICE)
        generate.assert_not_called()
        self.assertFalse(asks_current_values('Сколько стоил билет в 2001 году?'))

    def test_generation_rejects_wrong_number_and_object_property(self):
        self.assertEqual(_valid_fact('Башня имеет высоту 999 метров.', 'The tower is 634 m tall.'), '')
        self.assertEqual(_valid_fact('Здесь находится единственная монолитная арка.', 'The dome is very large.'), '')
        self.assertEqual(_valid_fact('Каждый час кукушка издаёт звук.', 'Every hour a rooster crows.'), '')
        self.assertEqual(_valid_fact('Это стеклянный шар, завершённый в 1999 году.', 'The glass dome was completed in 1999.'), '')

    def test_complex_source_is_reduced_to_requested_physical_facts(self):
        quote = 'The dial is a mechanical astrolabe, showing the current time, the placement of the sun and moon, and phases of the moon.'
        self.assertEqual(_prepared_fact(quote), 'Циферблат показывает: текущее время, положение Солнца и Луны, фазы Луны.')
        self.assertEqual(_prepared_fact('A slow-moving 12-month calendar with 19th century paintings.'), 'В механизме есть календарь на 12 месяцев.')

    def test_negated_property_cannot_become_positive(self):
        quote = 'This building has no glass dome.'
        self.assertEqual(_prepared_fact(quote), '')
        self.assertEqual(_valid_fact('У здания есть стеклянный купол.', quote), '')

    def test_russian_paraphrase_cannot_reverse_a_fact(self):
        facts = [passage('В механизме есть календарь на 12 месяцев.')]
        generate = Mock(return_value={'1': 'В механизме нет календаря на 12 месяцев.'})
        answer = answer_reference('Что есть в механизме?', facts, generate)
        self.assertIn('есть календарь', answer)
        self.assertNotIn('нет календаря', answer)

    def test_design_date_cannot_become_construction_date(self):
        quote = 'The gate was designed by Ada Stone in 1791.'
        self.assertEqual(_valid_fact('Ворота были построены в 1791 году.', quote), '')
        self.assertTrue(_valid_fact('Проект ворот датирован 1791 годом.', quote))

    def test_military_purpose_cannot_be_invented(self):
        quote = 'The gate is a symbol of the city.'
        self.assertEqual(_valid_fact('Ворота предназначались для безопасности и контроля города.', quote), '')

    def test_uncertain_claim_and_bc_era_must_survive(self):
        self.assertEqual(_valid_fact('Бронза использовалась для создания скульптур.',
                                     'The bronze might have been used for sculptures.'), '')
        self.assertEqual(_valid_fact('Мост был построен в 180 году нашей эры.',
                                     'The bridge was built in 180 BC.'), '')

    def test_each_fact_has_a_valid_citation_and_urls_are_deduplicated(self):
        facts = [passage('The tower is 634 m tall.'), passage('The tower was completed in 2000.')]
        generate = Mock(return_value={'1': 'Высота башни составляет 634 метра.',
                                      '2': 'Строительство башни завершено в 2000 году.'})
        answer = answer_reference('Расскажи о башне', facts, generate)
        self.assertEqual(answer.count('[Источник 1]'), 2)
        self.assertNotIn('[Источник 2]', answer)
        self.assertEqual(len(reference_sources(facts)), 1)

    def test_failed_generation_fails_closed(self):
        generate = Mock(return_value={'1': 'Построен в 9999 году.'})
        answer = answer_reference('Когда построен мост?', [passage('The bridge was built in 1500.')], generate)
        self.assertNotIn('9999', answer)
        self.assertIn('1500', answer)
        self.assertEqual(answer_reference('Когда построен мост?', [passage('The bridge can be found downtown.')], generate), NO_ANSWER)
