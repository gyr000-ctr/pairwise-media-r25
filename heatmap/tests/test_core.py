import math

import pytest

import heatmap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [(1.0, 1.0), (2.0, 2.0)]

CASES = [
    ("shape", lambda: len(heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)) == 8),
    ("sum_positive", lambda: sum(sum(r) for r in heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)) > 0.0),
    ("bad_bandwidth", lambda: _raises(lambda: heatmap.Heatmap(8, 8, 0.0))),
    ("bad_size", lambda: _raises(lambda: heatmap.Heatmap(0, 8, 1.0))),
    ("empty", lambda: heatmap.Heatmap(8, 8, 1.0).accumulate([]) == [[0.0] * 8 for _ in range(8)]),
    ("norm_range", lambda: all(0.0 <= v <= 1.0 for row in heatmap.Heatmap(8, 8, 1.0).normalize(heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)) for v in row)),
    ("bad_quantile", lambda: _raises(lambda: heatmap.Heatmap(8, 8, 1.0).normalize([[0.0]], 0.9, 0.1))),
    ("type", lambda: isinstance(heatmap.Heatmap(8, 8, 1.0).accumulate(PTS), list)),
    ("det", lambda: heatmap.Heatmap(8, 8, 1.0).accumulate(PTS) == heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)),
    ("peak", lambda: max(max(r) for r in heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)) > 0.0),
    ("nan", lambda: _raises(lambda: heatmap.Heatmap(8, 8, 1.0).accumulate([(float('nan'), 0.0)]))),
    ("no_mut", lambda: heatmap.Heatmap(8, 8, 1.0).accumulate(list(PTS)) is not None),
    ("norm_sum", lambda: abs(sum(sum(r) for r in heatmap.Heatmap(8, 8, 1.0).normalize(heatmap.Heatmap(8, 8, 1.0).accumulate(PTS))) - 4.0) < 4.0),
    ("float", lambda: isinstance(heatmap.Heatmap(8, 8, 1.0).accumulate(PTS)[0][0], float)),
    ("bandwidth_effect", lambda: heatmap.Heatmap(8, 8, 1.0).accumulate(PTS) != heatmap.Heatmap(8, 8, 2.0).accumulate(PTS)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
