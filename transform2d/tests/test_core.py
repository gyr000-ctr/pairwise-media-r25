import math

import pytest

import transform2d

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

M = [[2.0, 0.0, 1.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]

CASES = [
    ("identity", lambda: transform2d.Scene2D().world('root') == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ("add_dup", lambda: _raises(lambda: (lambda s: (s.add('a'), s.add('a')))(transform2d.Scene2D()))),
    ("cycle", lambda: _raises(lambda: (lambda s: (s.add('a'), s.add('b', 'a'), s.add('a', 'b')))(transform2d.Scene2D()))),
    ("missing", lambda: _raises(lambda: transform2d.Scene2D().world('zz'))),
    ("set_world", lambda: (lambda s: (s.set_local('root', M), s.world('root')))(transform2d.Scene2D()) is not None),
    ("inverse", lambda: isinstance(transform2d.Scene2D().inverse(M), list)),
    ("bad_inverse", lambda: _raises(lambda: transform2d.Scene2D().inverse([[1, 2, 0], [2, 4, 0], [0, 0, 1]]))),
    ("compose", lambda: isinstance(transform2d.Scene2D().compose(M, M), list)),
    ("type", lambda: isinstance(transform2d.Scene2D().world('root'), list)),
    ("det", lambda: transform2d.Scene2D().world('root') == transform2d.Scene2D().world('root')),
    ("depth_limit", lambda: _raises(lambda: transform2d.Scene2D(1).add('a', 'root'))),
    ("nan", lambda: _raises(lambda: transform2d.Scene2D().set_local('a', [[float('nan'), 0, 0], [0, 1, 0], [0, 0, 1]]))),
    ("no_mut", lambda: transform2d.Scene2D().compose(M, M) is not None),
    ("compose_identity", lambda: transform2d.Scene2D().compose(M, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])[0][2] == 1.0),
    ("inverse_type", lambda: isinstance(transform2d.Scene2D().inverse(M), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
