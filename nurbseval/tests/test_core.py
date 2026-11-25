import math

import pytest

import nurbseval

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

CP = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
KN = [0, 0, 0, 0, 1, 1, 1, 1]
W = [1.0, 1.0, 1.0, 1.0]

CASES = [
    ("start", lambda: abs(nurbseval.Nurbs(3, CP, KN, W).eval(0.0)[0]) < 1e-9),
    ("end", lambda: abs(nurbseval.Nurbs(3, CP, KN, W).eval(1.0)[0] - 3.0) < 1e-6),
    ("sum_basis", lambda: abs(sum(nurbseval.Nurbs(3, CP, KN, W).basis(0.5)) - 1.0) < 1e-12),
    ("bad_knots", lambda: _raises(lambda: nurbseval.Nurbs(3, CP, [0, 0, 1, 1], W))),
    ("bad_degree", lambda: _raises(lambda: nurbseval.Nurbs(0, CP, KN, W))),
    ("zero_weights", lambda: _raises(lambda: nurbseval.Nurbs(3, CP, KN, [0.0] * 4))),
    ("bad_t", lambda: _raises(lambda: nurbseval.Nurbs(3, CP, KN, W).eval(1.5))),
    ("deriv", lambda: isinstance(nurbseval.Nurbs(3, CP, KN, W).derivative(0.5), tuple)),
    ("type", lambda: isinstance(nurbseval.Nurbs(3, CP, KN, W).eval(0.5), tuple)),
    ("det", lambda: nurbseval.Nurbs(3, CP, KN, W).eval(0.5) == nurbseval.Nurbs(3, CP, KN, W).eval(0.5)),
    ("monotone", lambda: nurbseval.Nurbs(3, CP, KN, W).eval(0.2)[0] <= nurbseval.Nurbs(3, CP, KN, W).eval(0.8)[0]),
    ("nan", lambda: _raises(lambda: nurbseval.Nurbs(3, CP, KN, W).eval(float('nan')))),
    ("no_mut", lambda: nurbseval.Nurbs(3, list(CP), list(KN), list(W)).eval(0.3) is not None),
    ("deriv_type", lambda: isinstance(nurbseval.Nurbs(3, CP, KN, W).derivative(0.2), tuple)),
    ("basis_type", lambda: isinstance(nurbseval.Nurbs(3, CP, KN, W).basis(0.5), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
