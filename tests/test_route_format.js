const assert = require('node:assert/strict');
const { parseAnswer } = require('../static/route_format.js');

const result = parseAnswer(`День 1
1. Edo-Tokyo Museum [Источник 2] — Музей истории города.
2. Sensō-ji [Источник 1] — Буддийский храм — крупнейший в городе.
3. Saint-Alexandre-Nevsky Cathedral [Источник 2] — Собор XIX века.
День 2
1. Example-Park — Парк с театром под открытым небом.
2. Alpha-Beta Museum [Источник 3]
Источники:
[Источник 1] Example: https://en.wikivoyage.org/wiki/Example
Порядок остановок сгруппирован приблизительно по разделам путеводителя.`);
assert.equal(result.kind, 'route');
assert.deepEqual(result.days[0].stops.map(s => s.name), [
  'Edo-Tokyo Museum', 'Sensō-ji', 'Saint-Alexandre-Nevsky Cathedral'
]);
assert.deepEqual(result.days[0].stops.map(s => s.source), [2, 1, 2]);
assert.equal(result.days[0].stops[1].description, 'Буддийский храм — крупнейший в городе.');
assert.equal(result.days[1].stops[0].name, 'Example-Park');
assert.equal(result.days[1].stops[0].source, null);
assert.equal(result.days[1].stops[1].name, 'Alpha-Beta Museum');
assert.equal(result.days[1].stops[1].description, '');
assert.ok(result.note.startsWith('Порядок остановок'));

const empty = parseAnswer('День 1\nПодтверждённых объектов для этого дня недостаточно.\nДень 2\nПодтверждённых объектов для этого дня недостаточно.');
assert.equal(empty.kind, 'route');
assert.equal(empty.days.length, 2);
assert.ok(empty.days.every(d => d.stops.length === 0 && d.message));
assert.equal(parseAnswer('Обычный справочный ответ.').kind, 'text');
console.log('Route card parser regressions passed.');
