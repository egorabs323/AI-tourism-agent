MODES = {
    'route': {
        'name': 'Планирование маршрутов',
        'search': 'sightseeing itinerary must-see attractions travel guide',
    },
    'recommendations': {
        'name': 'Рекомендации',
        'search': 'recommended places sights attractions things to see and do',
    },
    'reference': {
        'name': 'Справочная информация',
        'search': '',
    },
}


def validate_mode(value):
    if value not in MODES:
        raise ValueError('Неизвестный тип запроса. Доступны: route, recommendations, reference')
    return value


def retrieval_query(question, mode):
    validate_mode(mode)
    suffix = MODES[mode]['search']
    return question if not suffix else '{} {}'.format(question, suffix)
