import unittest
from request_router import classify_request


class RouterTests(unittest.TestCase):
    def test_route_planning(self):
        self.assertEqual(classify_request('Составь туристический маршрут по Риму на два дня.'), 'route')
        self.assertEqual(classify_request('Построй маршрут по Казани за 2 дня'), 'route')
        self.assertEqual(classify_request('Предложи план поездки на выходные'), 'route')

    def test_ordinary_questions_do_not_trigger_route_builder(self):
        self.assertEqual(classify_request('Где начинается маршрут «Лазурная тропа»?'), 'qa')
        self.assertEqual(classify_request('А где он заканчивается?'), 'qa')
        self.assertEqual(classify_request('Сколько времени занимает маршрут?'), 'qa')
        self.assertEqual(classify_request('Какие главные достопримечательности есть в Риме?'), 'qa')
        self.assertEqual(classify_request('Какие из них лучше посетить с детьми?'), 'qa')
        self.assertEqual(classify_request('А какие из этих мест бесплатные?'), 'qa')

    def test_edits_are_not_new_route_creation(self):
        self.assertEqual(classify_request('Измени маршрут так, чтобы в первый день были только исторические достопримечательности.'), 'route_edit')
        self.assertEqual(classify_request('А теперь убери из первого дня музеи.'), 'route_edit')

if __name__ == '__main__':
    unittest.main()
