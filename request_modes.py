"""Selected user task for local Wikivoyage RAG. Python 3.9+; no new models."""

MODES = {
    'route': {
        'name': 'Планирование маршрута',
        'search': 'sightseeing itinerary must-see attractions travel guide',
    },
    'recommendations': {
        'name': 'Рекомендации мест',
        'search': 'recommended places sights attractions things to see and do',
    },
    'reference': {
        'name': 'Справочная информация',
        'search': '',
    },
    'transport': {
        'name': 'Транспорт и передвижение',
        'search': 'get around transport metro train bus travel connections',
    },
    'food': {
        'name': 'Еда и местная кухня',
        'search': 'eat food local cuisine dishes restaurants',
    },
    'accommodation': {
        'name': 'Проживание',
        'search': 'sleep stay accommodation hotels hostels neighbourhood',
    },
    'tips': {
        'name': 'Советы путешественникам',
        'search': 'practical travel advice safety customs etiquette money',
    },
}


def validate_mode(value):
    if value not in MODES:
        raise ValueError('Неизвестный тип запроса')
    return value


def retrieval_query(question, mode):
    validate_mode(mode)
    suffix = MODES[mode]['search']
    return question if not suffix else '{} {}'.format(question, suffix)
