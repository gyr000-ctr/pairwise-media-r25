import math

import pytest

import normalblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

N = (0.0, 0.0, 1.0)
T = (1.0, 0.0, 0.0)

CASES = [
    ("unit", lambda: abs(math.sqrt(sum(c * c for c in normalblend.NormalBlend().blend(N, N))) - 1.0) < 1e-9),
    ("tbn_orth", lambda: abs(sum(a * b for a, b in zip(normalblend.NormalBlend().tbn(N, T)[0], normalblend.NormalBlend().tbn(N, T)[1]))) < 1e-9),
    ("identity", lambda: all(abs(a - b) < 1e-9 for a, b in zip(normalblend.NormalBlend().blend(N, N), N))),
    ("bad_n", lambda: _raises(lambda: normalblend.NormalBlend().blend((0.0, 0.0), N))),
    ("zero_n", lambda: _raises(lambda: normalblend.NormalBlend().blend((0.0, 0.0, 0.0), N))),
    ("bad_mode", lambda: _raises(lambda: normalblend.NormalBlend().blend(N, N, 'bogus'))),
    ("whiteout", lambda: isinstance(normalblend.NormalBlend().blend(N, N, 'whiteout'), tuple)),
    ("type", lambda: isinstance(normalblend.NormalBlend().blend(N, N), tuple)),
    ("det", lambda: normalblend.NormalBlend().blend(N, N) == normalblend.NormalBlend().blend(N, N)),
    ("tbn_type", lambda: isinstance(normalblend.NormalBlend().tbn(N, T), tuple)),
    ("range", lambda: all(-1.0 <= v <= 1.0 for v in normalblend.NormalBlend().blend((0.3, 0.3, 0.9), (0.1, 0.2, 0.97)))),
    ("nan", lambda: _raises(lambda: normalblend.NormalBlend().blend((float('nan'), 0, 1), N))),
    ("no_mut", lambda: normalblend.NormalBlend().blend(N, N) is not None),
    ("tbn_unit", lambda: abs(math.sqrt(sum(c * c for c in normalblend.NormalBlend().tbn(N, T)[2])) - 1.0) < 1e-9),
    ("blend_len", lambda: len(normalblend.NormalBlend().blend(N, N)) == 3),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
