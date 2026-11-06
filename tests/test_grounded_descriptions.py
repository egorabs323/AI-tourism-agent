import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rag_pipeline import ArticleStore, Passage
from structured_routes import object_facts, city_catalogue, answer_route, extract_route_candidates, evidence_options
from route_enrichment import _batch_translate, _translation_valid, _source_summary


class GroundedDescriptionsTests(unittest.TestCase):
    def test_caption_before_unrelated_listing_is_not_ownership(self):
        text = ("Tower of London and Tower Bridge * £16.50 The great domed cathedral of "
                "St Paul's, designed by Sir Christopher Wren, was built in the 17th century.")
        self.assertEqual(object_facts(text, 'Tower Bridge'), [])

    def test_transport_and_nearby_hotel_are_not_descriptions(self):
        text = ('* Near Green Park. One of the great hotels built in the 19th century. '
                '* Tower Hill station is near Tower Bridge. Take the tube to the museum.')
        self.assertEqual(object_facts(text, 'Green Park'), [])
        self.assertEqual(object_facts(text, 'Tower Hill'), [])

    def test_record_primary_type_cannot_be_taken_from_neighbour(self):
        text = ('* Tower Bridge. The iconic 19th century bridge located by the Tower of London. '
                'It is decorated with high towers featuring a drawbridge. (Tube: Tower Hill)')
        self.assertEqual(object_facts(text, 'Tower Hill'), [])
        self.assertTrue(object_facts(text, 'Tower Bridge'))

    def test_compound_sentence_stops_before_second_object(self):
        text = ('Alpha Gallery houses a private art collection, Beta Museum houses '
                'a collection of archaeological artefacts.')
        facts = object_facts(text, 'Alpha Gallery')
        self.assertTrue(facts)
        self.assertNotIn('Beta', facts[0][1])

    def test_photo_metadata_is_removed_before_subject_detection(self):
        text = ('* Notre Dame de Paris 2013-07-24.jpg The early Gothic Cathédrale de '
                "Notre Dame (Our Lady) has a 12th century design but wasn't completed until the 14th.")
        facts = object_facts(text, 'Cathédrale de Notre Dame')
        self.assertTrue(facts)
        self.assertNotIn('.jpg', facts[0][1])

    def test_measurement_clause_and_complete_fact(self):
        text = '50 m high and 45 m wide, the Alpha Arch is decorated with battle scenes and martial sculptures.'
        facts = object_facts(text, 'Alpha Arch')
        self.assertTrue(facts)
        self.assertIn('decorated', facts[0][1])

    def test_no_cross_bullet_extension(self):
        text = ('* Alpha Palace is a baroque palace built in the 17th century. '
                '* Beta Church contains a collection of ancient sculptures.')
        facts = object_facts(text, 'Alpha Palace')
        self.assertTrue(facts)
        self.assertTrue(all('Beta' not in quote for _, quote in facts))

    def test_district_article_is_not_stop(self):
        p = Passage('North District is a district built in the 19th century. '
                    'Alpha Palace is a baroque palace built in the 17th century.',
                    'City/North District', 'https://en.wikivoyage.org/wiki/City/North_District', 0, '1')
        self.assertNotIn('North District', [a.name for a in city_catalogue([p], 'City')])

    def test_names_are_selected_from_records_without_llm(self):
        p = Passage('Alpha Palace is a baroque palace built in the 17th century.',
                    'City', 'https://en.wikivoyage.org/wiki/City', 0, '1')
        with patch('structured_routes._ollama_json', side_effect=AssertionError('selection must be local')):
            found, _ = extract_route_candidates([p], 'City', 2, None)
        self.assertEqual([a.name for a in found], ['Alpha Palace'])

    def test_empty_route_keeps_requested_days(self):
        output = answer_route('Маршрут на два дня', [], 'City')
        self.assertIn('День 1', output)
        self.assertIn('День 2', output)
        self.assertIn('недостаточно', output)

    def test_translation_rejects_new_numbers_and_wrong_object_type(self):
        quotes = {'Alpha Palace': 'Alpha Palace was built in 1700.',
                  'Beta Church': 'Beta Church was built in 1800.'}
        def generator(system, user, model, tokens):
            return {'text': 'Это церковь, построенная в 1900 году.'}
        with patch('route_enrichment._source_summary', return_value=''):
            self.assertEqual(_batch_translate(quotes, generator=generator), {})
        self.assertEqual(_translation_valid('Это парк, построенный в 1700 году.',
                         'Alpha Palace was built in 1700.', 'Alpha Palace'), '')

    def test_bc_cannot_turn_into_ad(self):
        self.assertEqual(_translation_valid('Мост был построен около 180 г. н. э.',
                         'The bridge was built around 180 BC.', 'Old Bridge'), '')

    def test_isolated_translation_preserves_source_mapping(self):
        quotes = {'Alpha Palace': 'Alpha Palace was built in 1700.',
                  'Beta Church': 'Beta Church was built in 1800.'}
        calls = []
        def generator(system, user, model, tokens):
            calls.append(json.loads(user))
            return {'text': 'Дворец был построен в 1700 году.' if json.loads(user)['object'] == 'Alpha Palace'
                    else 'Церковь была построена в 1800 году.'}
        with patch('route_enrichment._source_summary', return_value=''):
            self.assertEqual(len(_batch_translate(quotes, generator=generator)), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c['object'] for c in calls], list(quotes))

    def test_source_facts_work_without_translation_or_city_rules(self):
        text = 'The Star Observatory when completed measured 24 m high, 60 m in length, and 30 m in width.'
        result = _source_summary('Star Observatory', text)
        self.assertIn('высота — 24 м', result)
        self.assertIn('длина — 60 м', result)
        self.assertEqual(_source_summary('Unknown Tower', 'Unknown Tower can be found downtown.'), '')

    def test_generic_district_and_transport_headings_not_selected(self):
        passages = [Passage('* Tube. At the western end of the Piazza Completed in 1633 to a design by Someone. '
                            '* St Pancras is a largely residential area with 19th-century houses and gardens. '
                            '* Alpha Palace is a baroque palace built in the 17th century.',
                            'City', 'https://en.wikivoyage.org/wiki/City', 0, '1')]
        names = [a.name for a in city_catalogue(passages, 'City')]
        self.assertNotIn('Tube', names)
        self.assertNotIn('St Pancras', names)

    def test_person_biography_not_selected_as_building(self):
        passage = Passage('Saint Victor is said to have founded the church with Saint George.',
                          'City', 'https://en.wikivoyage.org/wiki/City', 0, '1')
        self.assertEqual(city_catalogue([passage], 'City'), [])

    def test_fact_relations_do_not_change_opening_or_art_into_construction(self):
        text = _source_summary('Test Museum', 'Test Museum was founded in 1700.')
        self.assertIn('основано', text)
        self.assertNotIn('построено', text)
        self.assertEqual(_source_summary('Test Gallery', 'A collection of Renaissance paintings.'), '')

    def test_material_and_designer_are_not_specific_to_a_city(self):
        text = _source_summary('White Monument', 'White Monument is a marble monument designed by Ada Stone.')
        self.assertIn('мрамор', text)
        self.assertIn('Ada Stone', text)

    def test_detached_heading_requires_matching_object_type(self):
        self.assertTrue(object_facts('See Alpha Palace * A royal palace built in the 17th century.', 'Alpha Palace'))
        self.assertEqual(object_facts('See Alpha Palace * A Buddhist temple built in 1500 with a large dome.', 'Alpha Palace'), [])

    def test_prose_references_cannot_become_detached_headings(self):
        for prefix in ['The garden is near Alpha Palace, imperial headquarters then',
                       'Given the proximity to Alpha Palace, imperial headquarters then',
                       'It’s smaller than Alpha Palace, but still worth visiting']:
            with self.subTest(prefix=prefix):
                self.assertEqual(object_facts('See ' + prefix + ' * A royal palace built in 1800.', 'Alpha Palace'), [])

    def test_single_word_foreign_subject_cannot_own_listing(self):
        text = 'See Alpha Park * Serpentine is a royal lake built in the 18th century.'
        self.assertEqual(object_facts(text, 'Alpha Park'), [])

    def test_coordinated_subject_does_not_transfer_shared_record(self):
        text = 'Alpha Mosque (Blue Mosque), and Beta Cistern, are decorated with ancient columns.'
        self.assertEqual(object_facts(text, 'Alpha Mosque'), [])

    def test_exact_name_identity_does_not_attach_suffix_institution(self):
        text = 'National Museum of Abc houses a collection of ancient sculptures.'
        self.assertTrue(object_facts(text, 'National Museum of Abc'))
        self.assertEqual(object_facts(text, 'Museum of Abc'), [])

    def test_parent_institution_reference_does_not_own_branch_description(self):
        text = ('See * branch of the National Museum of Abc (Museo Nazionale) '
                'access to all four National Museum of Abc sites Branch Excavation.jpg '
                'This museum is built on top of an ancient excavation site.')
        self.assertEqual(object_facts(text, 'National Museum of Abc'), [])

    def test_bilingual_listing_and_native_parenthetical_definition(self):
        text = ('See * Museums Alpha Museum 西洋美術館 9:00-17:00. '
                'This museum houses almost 100,000 art objects.')
        facts = object_facts(text, 'Alpha Museum')
        self.assertTrue(facts)
        self.assertNotIn('9:00', facts[0][1])
        text = 'Kōzan-ji (高山寺) is a Buddhist temple built in 1200 with a large collection of art.'
        self.assertTrue(object_facts(text, 'Kōzan-ji'))

    def test_unsupported_first_sentence_cannot_skip_to_another_object(self):
        text = ('See Alpha Palace * It opened with much fanfare. '
                'Beta Museum houses a collection of ancient art.')
        self.assertEqual(object_facts(text, 'Alpha Palace'), [])

    def test_hotel_and_sentence_adverb_are_not_route_stops(self):
        text = ('See Alpha Palace is a royal palace built in 1700. '
                'Downstairs is a small museum with a collection of ancient art. '
                'Walk downstairs and explore downstairs. '
                'Sleep Park Example Hotel is a royal residence built in 1800.')
        p = Passage(text, 'City', 'https://en.wikivoyage.org/wiki/City', 0, '1')
        self.assertEqual([a.name for a in city_catalogue([p], 'City')], ['Alpha Palace'])

    def test_untyped_district_and_closed_record_are_not_usable_evidence(self):
        text = ('West Quarter is an ancient district with a collection of museums. '
                'Alpha Museum is an archaeological museum closed for renovation until 2025.')
        p = Passage(text, 'City', 'https://en.wikivoyage.org/wiki/City', 0, '1')
        self.assertEqual(evidence_options([p], 'West Quarter'), [])
        self.assertEqual(evidence_options([p], 'Alpha Museum'), [])

    def test_overview_prominence_outweighs_minor_district_listing(self):
        passages = [Passage('See Alpha Mosque is a mosque built in 1600. Alpha Mosque is a royal mosque.',
                            'City', 'https://en.wikivoyage.org/wiki/City', 0, '1'),
                    Passage('See Beta Church * A church built in 1500 with ancient sculptures.',
                            'City/Island', 'https://en.wikivoyage.org/wiki/City/Island', 0, '2')]
        self.assertEqual(city_catalogue(passages, 'City')[0].name, 'Alpha Mosque')

    def test_short_fact_summaries_preserve_dates_era_and_quantities(self):
        cases = [
            ('Completed in 537 AD, this is an ancient building.', '537 году'),
            ('The shrine was built in the 4th century BC.', 'IV веке до н. э.'),
            ('This museum houses almost 100,000 art objects.', 'почти 100 000 предметов искусства'),
            ('The mosque has a great dome and six minarets.', '6 минаретов и большой купол'),
        ]
        for quote, expected in cases:
            with self.subTest(quote=quote):
                self.assertIn(expected, _source_summary('Example', quote))

    def test_market_relocation_qualifier_survives_short_description(self):
        text = ("Central Market was the world's largest wholesale fish market. "
                'Although the wholesale operations moved, the outer market remains in place to welcome the public.')
        fact = object_facts(text, 'Central Market')[0][1]
        summary = _source_summary('Central Market', fact)
        self.assertIn('Ранее был', summary)
        self.assertIn('Внешний рынок продолжает принимать посетителей', summary)


class ArticleStoreTests(unittest.TestCase):
    def test_city_scope_byte_offsets_and_invalidation(self):
        with TemporaryDirectory() as folder:
            path, cache = Path(folder) / 'articles.jsonl', Path(folder) / 'offsets.sqlite3'
            rows = [dict(article_id=str(i), title=title, url='https://en.wikivoyage.org/wiki/' + title,
                         text='Данные ' + title)
                    for i, title in enumerate(['City', 'City/North', 'Cityscape', 'Other'])]
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            store = ArticleStore(path, cache)
            self.assertEqual([p.title for p in store.city('City')], ['City', 'City/North'])
            with patch.object(store, '_connect', side_effect=AssertionError('cache must be used')):
                self.assertEqual(len(store.city('City')), 2)
            rows[0]['text'] = 'Updated article text with more content'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            self.assertEqual(store.city('City')[0].text, rows[0]['text'])
