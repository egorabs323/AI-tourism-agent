import math
import unittest
from geo_routes import haversine_km, geographic_groups
from structured_routes import Attraction

class GeoRoutesTests(unittest.TestCase):
    def make(self, name, source=1):
        return Attraction(name, source, 'Rome', 'https://example.org', name + ' is here')

    def test_distance_same_point_is_zero(self):
        self.assertEqual(haversine_km((41.89, 12.49), (41.89, 12.49)), 0)

    def test_rome_cluster_only_verified_objects(self):
        attractions = [self.make(name, i) for i, name in enumerate(
            ['Colosseum', 'Pantheon', 'Sistine Chapel', 'Fontana dell\'Acqua Paola',
             'San Pietro in Montorio', 'Palazzo Farnese'], 1)]
        result = geographic_groups(attractions, 'Rome', 2)
        self.assertIsNotNone(result)
        groups, note = result
        self.assertEqual([len(g) for g in groups], [3, 3])
        self.assertEqual(set(a.name for day in groups for a in day), set(a.name for a in attractions))
        self.assertIn('по прямой', note)

    def test_no_city_coordinates_substitute(self):
        result = geographic_groups([self.make('Unrecognized Site')], 'Rome', 1)
        self.assertIsNone(result)

    def test_other_city_falls_back(self):
        self.assertIsNone(geographic_groups([self.make('Eiffel Tower')], 'Paris', 1))

    def test_excludes_unverified_place(self):
        result = geographic_groups([self.make('Colosseum'), self.make('Fake Castle'),
                                    self.make('Pantheon')], 'Rome', 1)
        self.assertIsNotNone(result)
        groups, note = result
        self.assertNotIn('Fake Castle', [a.name for d in groups for a in d])
        self.assertIn('Не включено', note)

if __name__ == '__main__':
    unittest.main()
