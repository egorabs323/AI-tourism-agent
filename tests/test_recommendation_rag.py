"""Recommendation regressions on synthetic sources, without live models."""
import json
import unittest
from unittest.mock import Mock

from rag_pipeline import Passage
from recommendation_rag import (RecommendationRequest, prepare_recommendations,
                                select_recommendations, answer_recommendations, NO_RECOMMENDATIONS)


def p(text, title='City', url='https://example.test/city', cid='1'):
    return Passage(text, title, url, .1, cid)


def request(kind, count=3, **kwargs):
    return RecommendationRequest(query=kind, kinds=frozenset([kind]), count=count, **kwargs)


class RecommendationTests(unittest.TestCase):
    def test_positive_type_is_not_taken_from_exclusion(self):
        generate = Mock(return_value={'query': 'two historic palaces; no churches or modern art museums'})
        r = prepare_recommendations('Порекомендуй два исторических дворца. Церкви и музеи современного искусства не включай.', [], 'City', generate)
        self.assertEqual(r.count, 2)
        self.assertEqual(r.kinds, {'palace'})
        self.assertEqual(r.excluded_kinds, {'church', 'museum'})
        self.assertFalse(r.art)

    def test_named_exception_does_not_exclude_all_museums(self):
        generate = Mock(return_value={'query': 'art museums except Alpha Museum', 'excluded_names': ['Alpha Museum']})
        r = prepare_recommendations('Порекомендуй три художественных музея, кроме Alpha Museum.', [], 'City', generate)
        self.assertEqual(r.kinds, {'museum'})
        self.assertNotIn('museum', r.excluded_kinds)
        self.assertIn('Alpha Museum', r.excluded_names)

    def test_conditions_in_one_sentence_do_not_erase_requested_type(self):
        generate = Mock(return_value={'query': 'two parks without museums'})
        r = prepare_recommendations('Порекомендуй два парка без музеев', [], 'City', generate)
        self.assertEqual(r.kinds, {'park'})
        self.assertEqual(r.excluded_kinds, {'museum'})
        generate.return_value = {'query': 'two art museums except Alpha Museum', 'excluded_names': ['Alpha Museum']}
        r = prepare_recommendations('Порекомендуй два художественных музея кроме Alpha Museum', [], 'City', generate)
        self.assertEqual(r.kinds, {'museum'})
        self.assertTrue(r.art)
        self.assertNotIn('museum', r.excluded_kinds)

    def test_query_keeps_interests_and_only_user_history(self):
        generate = Mock(return_value={'query': 'parks and gardens, no museums'})
        history = [{'role': 'user', 'content': 'Нравятся парки'}, {'role': 'assistant', 'content': 'Недостоверное описание.'}]
        r = prepare_recommendations('Порекомендуй два парка, без музеев.', history, 'City', generate)
        self.assertIn('parks and gardens', r.query)
        self.assertNotIn('Недостоверное', generate.call_args.args[1])

    def test_followup_inherits_type_but_new_type_overrides_it(self):
        history = [{'role': 'user', 'content': 'Порекомендуй три художественных музея.'}]
        generate = Mock(return_value={'query': 'more museums'})
        self.assertEqual(prepare_recommendations('Ещё два.', history, 'City', generate).kinds, {'museum'})
        self.assertEqual(prepare_recommendations('А теперь два парка.', history, 'City', generate).kinds, {'park'})
        self.assertTrue(prepare_recommendations('Посоветуй ещё один такой музей.', history, 'City', generate).followup)

    def test_types_cannot_turn_a_cathedral_or_street_into_a_palace_or_park(self):
        sources = [p('See * Alpha Cathedral.jpg This Gothic cathedral was built in 1700. '
                     '* Beta Street is a shopping street with galleries and cafés.')]
        self.assertEqual(select_recommendations(request('palace'), sources), [])
        self.assertEqual(select_recommendations(request('park'), sources), [])

    def test_sentence_pronouns_and_transport_ids_are_not_names(self):
        sources = [p('See * Adults Bus X33 Photo.jpg A museum of modern art. '
                     '* This Baroque palace was built in 1700. '
                     '* There is also a gallery of contemporary art.')]
        for kind in ['museum', 'palace']:
            self.assertEqual(select_recommendations(request(kind), sources), [])

    def test_compound_museum_name_is_supported(self):
        sources = [p('See * StarMuseum is a museum with a collection of paintings.')]
        places = select_recommendations(request('museum', art=True), sources)
        self.assertEqual([v.name for v in places], ['StarMuseum'])

    def test_short_collection_description_survives_fact_validation(self):
        source = p('See Alpha Museum is a museum with a sculpture collection.')
        answer, _ = answer_recommendations('Один музей', [source], [], request('museum', 1),
                                           generator=Mock(return_value={'1': 'Скульптура.'}))
        self.assertIn('Alpha Museum', answer)
        self.assertIn('скульптура', answer.lower())

    def test_city_possessive_district_is_not_a_museum(self):
        source = p("See This is City's museum district, named for three major art museums.")
        self.assertEqual(select_recommendations(request('museum', art=True), [source]), [])

    def test_native_name_preserves_city_after_preposition(self):
        source = p('See * Museo de Historia de City.jpg This museum contains paintings.')
        self.assertEqual([v.name for v in select_recommendations(request('museum'), [source])],
                         ['Museo de Historia de City'])

    def test_typed_caption_owns_an_art_collection_definition(self):
        source = p('See * Street Alpha, 22 Museo del Example.jpg One of the finest art collections in the region.')
        names = [v.name for v in select_recommendations(request('museum', art=True), [source])]
        self.assertIn('Museo del Example', names)
        self.assertNotIn('Street Alpha', names)

    def test_listing_heading_can_own_a_matching_park_definition(self):
        source = p('See Parc Beta * Street Alpha 362 Urban park created from redeveloping industrial land. '
                   'It covers seven hectares and includes a lake with sculptures.')
        places = select_recommendations(request('park'), [source])
        self.assertEqual([v.name for v in places], ['Parc Beta'])

    def test_park_does_not_borrow_a_nearby_museum_description(self):
        sources = [p('See * Near Alpha Park. Beta Museum.jpg This museum houses paintings and sculptures.')]
        self.assertEqual(select_recommendations(request('park'), sources), [])

    def test_photo_metadata_separates_nearby_museum_and_actual_gallery(self):
        sources = [p('See * Reduced admission with a ticket from the nearby Alpha Museum Beta Galerie aussen.jpg '
                     'A museum of modern art and design.')]
        places = select_recommendations(request('museum', art=True), sources)
        self.assertEqual([v.name for v in places], ['Beta Galerie'])

    def test_parent_museum_cannot_own_branch_description(self):
        sources = [p('See * branch of the National Museum of Abc, '
                     'access to all four National Museum of Abc sites Branch Photo.jpg '
                     'This museum houses a collection of ancient paintings.')]
        places = select_recommendations(request('museum', art=True), sources)
        self.assertNotIn('National Museum of Abc', [v.name for v in places])

    def test_distinct_named_subjects_keep_separate_collections(self):
        sources = [p('See Alpha Museum houses a collection of paintings, '
                     'while Beta Museum houses a collection of sculptures.')]
        places = select_recommendations(request('museum', art=True), sources)
        for place in places:
            if place.name == 'Alpha Museum':
                self.assertNotIn('sculptures', place.evidence)
            if place.name == 'Beta Museum':
                self.assertNotIn('paintings', place.evidence)
        self.assertEqual({v.name for v in places}, {'Alpha Museum', 'Beta Museum'})

    def test_explicit_parenthetical_aliases_count_as_one_place(self):
        sources = [p('See * Schloss Alpha (Alpha Palace).jpg This Baroque palace was built in 1700. '
                     '* Alpha Palace is a palace with a royal collection.')]
        places = select_recommendations(request('palace'), sources)
        self.assertEqual(len(places), 1)

    def test_industrial_park_and_carriage_gallery_are_not_nature_parks(self):
        sources = [p('See * Alpha Technology Park.jpg A business park with offices and industrial companies. '
                     '* Beta Park Carriages.jpg A collection of royal carriages for viewing.')]
        self.assertEqual(select_recommendations(request('park'), sources), [])

    def test_excluded_place_does_not_appear(self):
        sources = [p('See Alpha Museum is an art museum with paintings. '
                     'Beta Museum is an art museum with sculptures.')]
        r = request('museum', art=True, excluded_names=('Alpha Museum',))
        self.assertEqual([v.name for v in select_recommendations(r, sources)], ['Beta Museum'])

    def test_strict_topic_requires_all_requested_properties(self):
        sources = [p('See Alpha Museum is a museum with a collection of meteorites. '
                     'Beta Museum is a museum of modern art.')]
        r = request('museum', strict=True, strict_topics=('lunar', 'meteorites'))
        self.assertEqual(select_recommendations(r, sources), [])

    def test_unresolved_strict_topic_refuses_to_substitute_other_museums(self):
        sources = [p('See Alpha Museum is a museum of modern art.')]
        r = request('museum', strict=True)
        self.assertEqual(answer_recommendations('Только музеи узкой тематики', sources, [], r), (NO_RECOMMENDATIONS, []))

    def test_free_condition_requires_explicit_admission_evidence(self):
        sources = [p('See Alpha Museum is a museum with paintings and free WiFi.')]
        self.assertEqual(select_recommendations(request('museum', free=True), sources), [])

    def test_fewer_places_are_reported_without_filler(self):
        sources = [p('See Alpha Museum is a museum of modern art.')]
        generate = Mock(return_value={'1': 'Современное искусство.'})
        answer, links = answer_recommendations('Три художественных музея', sources, [], request('museum', art=True), generator=generate)
        self.assertIn('1. Alpha Museum', answer)
        self.assertNotIn('2.', answer)
        self.assertIn('1 из 3', answer)
        self.assertEqual(len(links), 1)

    def test_generator_cannot_add_names_or_wrong_numbers(self):
        sources = [p('See * Alpha Palace.jpg This Baroque palace was built in 1700.')]
        generate = Mock(return_value={'1': 'Построен в 9999 году.', '2': 'Invented Palace'})
        answer, _ = answer_recommendations('Один дворец', sources, [], request('palace', 1), generator=generate)
        self.assertIn('1700', answer)
        self.assertNotIn('9999', answer)
        self.assertNotIn('Invented', answer)

    def test_source_numbers_have_no_gaps_or_duplicate_urls(self):
        sources = [p('See Alpha Museum is a museum of modern art. Beta Museum is a museum of contemporary art.')]
        generate = Mock(return_value={'1': 'Современное искусство.', '2': 'Современное искусство.'})
        answer, links = answer_recommendations('Два музея', sources, [], request('museum', 2), generator=generate)
        self.assertEqual(len(links), 1)
        self.assertEqual(answer.count('[Источник 1]'), 2)
        self.assertNotIn('День ', answer)

    def test_russian_document_does_not_mix_museum_and_park(self):
        source = p('Музей «Северная палитра» — это художественный музей. '
                   'В коллекции представлены картины и рисунки местных художников.\n'
                   'Парк «Тихий луг» — это парк с озером и прогулочными дорожками.',
                   'Документ: guide.txt', 'http://localhost/document', 'doc_1')
        places = select_recommendations(request('museum', art=True), [source])
        self.assertEqual([v.name for v in places], ['Северная палитра'])
        self.assertTrue(all('Тихий луг' not in v.evidence for v in places))
        places = select_recommendations(request('park', 1), [source])
        self.assertEqual([v.name for v in places], ['Тихий луг'])
        self.assertTrue(all('Северная палитра' not in v.evidence for v in places))

    def test_followup_avoids_previously_recommended_names(self):
        sources = [p('See Alpha Museum is a museum of modern art. Beta Museum is a museum of contemporary art.')]
        history = [{'role': 'assistant', 'content': '1. Alpha Museum — недостоверный старый факт.'}]
        places = select_recommendations(request('museum', followup=True), sources, history=history)
        self.assertEqual([v.name for v in places], ['Beta Museum'])
