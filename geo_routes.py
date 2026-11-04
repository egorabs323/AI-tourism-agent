"""Lightweight, offline geographic clustering of verified attractions.

Coordinates are approximate reference points for individual attractions,
not pedestrian routing. Never substitute a city's coordinates for a place.
Only names already verified against Wikivoyage passages can be routed.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_COORDINATES = Path(__file__).with_name('data') / 'landmark_coordinates.json'


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between (latitude, longitude) points, in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(h)))


def load_coordinates(path: Path = DEFAULT_COORDINATES) -> dict:
    """Dictionary of cities -> exact attraction names -> coordinates and priorities."""
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            return {}
        return document
    except (OSError, ValueError):
        return {}


def lookup(city: str, name: str, registry: dict) -> dict | None:
    sites = registry.get(city.casefold(), {})
    if not isinstance(sites, dict):
        return None
    item = sites.get(name.casefold())
    if not isinstance(item, dict):
        return None
    lat, lon = item.get('lat'), item.get('lon')
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and
            not isinstance(lat, bool) and not isinstance(lon, bool) and
            -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return item


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(haversine_km(a, b) for a, b in zip(points, points[1:]))


def _best_order(indices: tuple[int, ...], coords: list[tuple[float, float]]) -> tuple[tuple[int, ...], float]:
    """Exact shortest straight-line order for up to 3 stops."""
    return min(
        ((order, _path_length([coords[i] for i in order])) for order in itertools.permutations(indices)),
        key=lambda item: (item[1], item[0])
    )


def geographic_groups(attractions: list[Any], destination: str, days: int,
                      per_day: int = 3, registry: dict | None = None
                      ) -> tuple[list[list[Any]], str] | None:
    """Select known, source-verified sites and group by approximate straight-line proximity.

    Returns None if too few attractions have individual coordinates; caller can
    retain the original non-geographical grouping in that case.
    """
    if days < 1 or days > 3 or per_day < 1 or per_day > 3:
        return None
    registry = registry if registry is not None else load_coordinates()
    known = []
    for attraction in attractions:
        item = lookup(destination, attraction.name, registry)
        if item is not None:
            known.append((attraction, item))
    if len(known) < days or not known:
        return None
    # Prefer popular landmarks explicitly enumerated in the local reference;
    # they MUST still have been extracted from retrieved documents.
    known.sort(key=lambda pair: (-float(pair[1].get('priority', 0)), pair[0].passage_index, pair[0].name))
    chosen = known[:days * per_day]
    stops = [a for a, _ in chosen]
    coords = [(item['lat'], item['lon']) for _, item in chosen]
    n = len(stops)
    # Enumerate small partitions of <=9 stops, each day nonempty / capped.
    # Fix first stop in day 0 to remove symmetric day permutations.
    # Objective: sum of per-day straight-line path lengths + small imbalance penalty.
    best = None
    for assignment_tail in itertools.product(range(days), repeat=n - 1):
        assignment = (0,) + assignment_tail
        buckets = [tuple(i for i, day in enumerate(assignment) if day == d) for d in range(days)]
        counts = [len(bucket) for bucket in buckets]
        if not all(1 <= count <= per_day for count in counts):
            continue
        ordered = [_best_order(bucket, coords) for bucket in buckets]
        lengths = [length for _, length in ordered]
        # Avoid one day with 1 stop and the other with 3, when possible.
        penalty = 0.35 * (max(counts) - min(counts))
        score = sum(lengths) + penalty
        candidate = (round(score, 8), assignment, ordered, lengths)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    _, _, ordered, lengths = best
    groups = [[stops[idx] for idx in order] for order, _ in ordered]
    # Qualify clearly: straight-line, not road/walking distance.
    note = (f'Группировка по координатам объектов: приблизительное расстояние по прямой '
            f'между остановками по дням: ' + '; '.join(
                f'день {day}: {length:.1f} км' for day, length in enumerate(lengths, 1)
            ) + '. Это не длина пешеходного маршрута и не время в пути.')
    missing = len(attractions) - len(chosen)
    if missing:
        note += f' Не включено объектов без координат или сверх лимита: {missing}.'
    return groups, note
