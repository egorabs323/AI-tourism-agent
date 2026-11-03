import unittest
from rag_pipeline import Passage
from structured_routes import validate_attractions, route_days, group_attractions, render_route

class RouteTests(unittest.TestCase):
    def setUp(self):
        self.p = [Passage('Visit Colosseum and Roman Forum today.', 'Rome/Colosseo', 'https://example.test/a', .1, '1'),
                  Passage('Pantheon is a wonderful historical monument.', 'Rome/Old Rome', 'https://example.test/b', .2, '2')]

    def test_days(self):
        self.assertEqual(route_days('Маршрут по Риму на два дня'), 2)
        self.assertEqual(route_days('Маршрут на 3 дня'), 3)

    def test_exact_names_only(self):
        actual = validate_attractions([{'name':'Colosseum', 'source':1},
                                       {'name':'Made-up Coliseum','source':1},
                                       {'name':'Pantheon','source':2}], self.p)
        self.assertEqual([a.name for a in actual], ['Colosseum','Pantheon'])

    def test_invalid_source(self):
        self.assertEqual(validate_attractions([{'name':'Pantheon','source':3}], self.p), [])

    def test_render_cannot_add_stops(self):
        a = validate_attractions([{'name':'Colosseum','source':1}], self.p)
        rendered = render_route(group_attractions(a, 2), {'1':'Римская достопримечательность.'}, a)
        self.assertIn('Colosseum', rendered)
        self.assertNotIn('Pantheon', rendered)
        self.assertIn('День 2', rendered)

if __name__=='__main__': unittest.main()
