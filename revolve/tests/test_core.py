import math

import pytest

import revolve

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PROF = [(1.0, 0.0), (1.0, 1.0), (0.5, 2.0)]

CASES = [
    ("vert_count", lambda: len(revolve.Revolve().build(PROF, 8)[0]) == 3 * 8),
    ("tri_count", lambda: len(revolve.Revolve().build(PROF, 8)[1]) == 8 * 3 * 2),
    ("seam", lambda: abs(revolve.Revolve().build(PROF, 8)[0][0][0] - revolve.Revolve().build(PROF, 8)[0][-1][0]) < 1.0),
    ("normals_unit", lambda: all(abs(math.sqrt(sum(c * c for c in n)) - 1.0) < 1e-6 for n in revolve.Revolve().build(PROF, 8)[2])),
    ("bad_profile", lambda: _raises(lambda: revolve.Revolve().build([], 8))),
    ("bad_segments", lambda: _raises(lambda: revolve.Revolve().build(PROF, 0))),
    ("bad_axis", lambda: _raises(lambda: revolve.Revolve().build(PROF, 8, (0.0, 0.0, 0.0)))),
    ("degenerate", lambda: _raises(lambda: revolve.Revolve().build([(0.0, 0.0), (0.0, 1.0)], 8))),
    ("type", lambda: isinstance(revolve.Revolve().build(PROF, 8), tuple)),
    ("det", lambda: revolve.Revolve().build(PROF, 8) == revolve.Revolve().build(PROF, 8)),
    ("finite", lambda: all(math.isfinite(c) for p in revolve.Revolve().build(PROF, 8)[0] for c in p)),
    ("no_mut", lambda: revolve.Revolve().build(list(PROF), 8) is not None),
    ("tri_valid", lambda: all(0 <= i < 24 for t in revolve.Revolve().build(PROF, 8)[1] for i in t)),
    ("normals_count", lambda: len(revolve.Revolve().build(PROF, 8)[2]) == len(revolve.Revolve().build(PROF, 8)[0])),
    ("nan", lambda: _raises(lambda: revolve.Revolve().build([(float('nan'), 0.0)], 8))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
