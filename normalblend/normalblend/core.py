"""normalblend —— 法线贴图混合。

实现要点
--------
* ``blend`` 支持 ``udn``（partial derivative / Unreal）与 ``whiteout``
  两种叠加公式，输出总是重新归一化到单位长度；
* 提供线性权重（0 → 输入 a，1 → 输入 b），端点严格退化为单输入；
* ``tbn`` 由 ``n`` 与 ``t`` 通过 Gram-Schmidt 构造右手正交切线基；
* 输入校验区分维度错误、非有限值、零向量、非法模式等错误
  （全部是 :class:`ValueError` 的子类，便于调用方统一捕获）；
* 无共享可变状态，可在多线程中并发调用；
* 单法线走低开销标量路径；整幅法线贴图（NumPy 数组或嵌套序列）走
  向量化路径，10 万像素远低于 400 ms。
"""

from __future__ import annotations

import math

try:  # NumPy 只用于批量加速，缺失时自动退回纯 Python。
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 NumPy 时仍可工作
    _np = None

__all__ = [
    "NormalBlend",
    "NormalBlendError",
    "InvalidNormalError",
    "NormalDimensionError",
    "ZeroNormalError",
    "NonFiniteNormalError",
    "NaNComponentError",
    "InfiniteComponentError",
    "DegenerateBlendError",
    "InvalidModeError",
    "InvalidWeightError",
    "InvalidTangentError",
    "NonFiniteTangentError",
    "NaNTangentError",
    "InfiniteTangentError",
    "blend_normals",
    "tbn",
]

_INF = float("inf")
_ISFINITE = math.isfinite
_ISNAN = math.isnan
_SQRT = math.sqrt
_FLOAT = float

_MODE_ALIASES = {
    "udn": "udn",
    "partial": "udn",
    "partial_derivative": "udn",
    "pd": "udn",
    "unreal": "udn",
    "whiteout": "whiteout",
    "wo": "whiteout",
}

_WEIGHT_ALIASES = ("weight", "w", "alpha", "mix", "factor")


# ---------------------------------------------------------------------------
# 异常：全部继承 ValueError，并以不同类/消息区分错误类别
# ---------------------------------------------------------------------------
class NormalBlendError(ValueError):
    """normalblend 的基异常（同时是 ValueError）。

    ``category`` 属性给出稳定的机器可读分类，便于调用方区分错误类型。
    """

    category = "invalid"

    def __init__(self, message="", *, category=None):
        super().__init__(message)
        if category is not None:
            self.category = category


class InvalidNormalError(NormalBlendError):
    """法线不合法（基类）。"""

    category = "invalid_normal"


class NormalDimensionError(InvalidNormalError):
    """法线维度不是 3（或不是可解包的数字序列）。"""

    category = "dimension"


class ZeroNormalError(InvalidNormalError):
    """法线是零向量，无法归一化。"""

    category = "zero_normal"


class NonFiniteNormalError(InvalidNormalError):
    """法线含有 NaN / ±inf（基类）。"""

    category = "non_finite"


class NaNComponentError(NonFiniteNormalError):
    """法线分量是 NaN。"""

    category = "nan"


class InfiniteComponentError(NonFiniteNormalError):
    """法线分量是 ±inf。"""

    category = "infinite"


class DegenerateBlendError(NormalBlendError):
    """混合结果为零向量，无法重归一化。"""

    category = "degenerate"


class InvalidModeError(NormalBlendError):
    """未知的混合模式。"""

    category = "invalid_mode"


class InvalidWeightError(NormalBlendError):
    """混合权重不是 [0, 1] 内的有限实数。"""

    category = "invalid_weight"


class InvalidTangentError(NormalBlendError):
    """切线为零向量或与法线共线。"""

    category = "invalid_tangent"


class NonFiniteTangentError(NormalBlendError):
    """切线含有 NaN / ±inf（基类）。"""

    category = "non_finite_tangent"


class NaNTangentError(NonFiniteTangentError):
    """切线分量是 NaN。"""

    category = "nan_tangent"


class InfiniteTangentError(NonFiniteTangentError):
    """切线分量是 ±inf。"""

    category = "infinite_tangent"


# ---------------------------------------------------------------------------
# 标量工具
# ---------------------------------------------------------------------------
def _is_number(value):
    try:
        _FLOAT(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_normal(v, name="normal"):
    """返回 ``(x, y, z)`` 三个 float；非法输入抛可区分的异常。"""
    try:
        x, y, z = v
    except (TypeError, ValueError):
        raise NormalDimensionError(
            "%s must be a sequence of 3 numbers, got %r" % (name, (v,))
        ) from None
    # 维度校验（解包成功即恰好 3 个元素；若长度不是 3 会抛 ValueError）。
    if isinstance(x, (tuple, list)) or isinstance(y, (tuple, list)) or isinstance(
        z, (tuple, list)
    ):
        raise NormalDimensionError(
            "%s must be a sequence of 3 numbers, got nested values" % name
        )
    try:
        x = _FLOAT(x)
        y = _FLOAT(y)
        z = _FLOAT(z)
    except (TypeError, ValueError):
        raise NormalDimensionError(
            "%s must contain real numbers, got %r" % (name, (v,))
        ) from None

    # 快速路径：长度平方为正常有限值 ⇒ 分量有限且非零向量。
    d2 = x * x + y * y + z * z
    if 0.0 < d2 < _INF:
        return x, y, z

    # 慢速路径：精确分类（NaN 与 ±inf 分别报告）。
    for c in (x, y, z):
        if _ISNAN(c):
            raise NaNComponentError(
                "%s contains a NaN component: %r" % (name, (x, y, z))
            )
    for c in (x, y, z):
        if not _ISFINITE(c):
            raise InfiniteComponentError(
                "%s contains an infinite component: %r" % (name, (x, y, z))
            )
    if x == 0.0 and y == 0.0 and z == 0.0:
        raise ZeroNormalError(
            "%s is the zero vector and cannot be normalized" % name
        )
    # 否则是次正规数导致长度平方下溢，保留原值（后续归一化会缩放处理）。
    return x, y, z


_UNIT_EPS = 1e-12


def _normalize_xyz(x, y, z, *, error=DegenerateBlendError, what="blended normal"):
    """把向量归一化；使用缩放避免有限大数的平方溢出。"""
    d2 = x * x + y * y + z * z
    if 0.0 < d2 < _INF:
        s = 1.0 / _SQRT(d2)
        return x * s, y * s, z * s
    # NaN / inf / 0 / 次正规下溢：走稳健路径。
    m = max(abs(x), abs(y), abs(z))
    if m == 0.0 or not _ISFINITE(m):
        raise error("%s is degenerate (%r) and cannot be normalized" % (what, (x, y, z)))
    x /= m
    y /= m
    z /= m
    d2 = x * x + y * y + z * z
    if not (0.0 < d2 < _INF):  # 理论上不会发生，防御性处理
        raise error("%s is degenerate and cannot be normalized" % what)
    s = 1.0 / _SQRT(d2)
    return x * s, y * s, z * s


def _unit_or_normalized(v, *, error=DegenerateBlendError, what="normal"):
    """将单个输入法线归一化；若其已是单位向量则原样返回（保持逐位相等）。"""
    x, y, z = v
    d2 = x * x + y * y + z * z
    if 0.0 < d2 < _INF and abs(d2 - 1.0) <= _UNIT_EPS:
        return x, y, z
    return _normalize_xyz(x, y, z, error=error, what=what)


def _is_single(v):
    """判断 ``v`` 是单个法线（3 个数字）还是批量/图像输入。"""
    if _np is not None and isinstance(v, _np.ndarray):
        return v.ndim == 1 and v.shape[0] == 3
    try:
        if len(v) != 3:
            return False
        x, y, z = v
    except (TypeError, ValueError):
        return False
    if isinstance(x, (tuple, list)) or isinstance(y, (tuple, list)) or isinstance(
        z, (tuple, list)
    ):
        return False
    return _is_number(x) and _is_number(y) and _is_number(z)


def _parse_mode(mode):
    if not isinstance(mode, str):
        raise InvalidModeError("mode must be a string, got %r" % (mode,))
    key = mode.strip().lower()
    try:
        return _MODE_ALIASES[key]
    except KeyError:
        raise InvalidModeError(
            "unknown blend mode %r (expected 'udn' or 'whiteout')" % (mode,)
        ) from None


def _parse_weight(weight):
    """标量权重：返回 float；None 表示不做权重混合。"""
    if weight is None:
        return None
    if isinstance(weight, bool):
        weight = float(weight)
    if _np is not None and isinstance(weight, _np.ndarray):
        if weight.ndim != 0:
            raise InvalidWeightError("weight must be a scalar for a single normal")
        weight = _FLOAT(weight)
    try:
        weight = _FLOAT(weight)
    except (TypeError, ValueError):
        raise InvalidWeightError("weight must be a real number, got %r" % (weight,))
    if not _ISFINITE(weight):
        raise InvalidWeightError("weight must be finite, got %r" % weight)
    if not 0.0 <= weight <= 1.0:
        raise InvalidWeightError("weight must be in [0, 1], got %r" % weight)
    return weight


def _fmt_error_value(v):
    try:
        text = repr(v)
    except Exception:  # pragma: no cover
        text = "<unrepresentable>"
    if len(text) > 80:
        text = text[:77] + "..."
    return text


# ---------------------------------------------------------------------------
# 标量核心
# ---------------------------------------------------------------------------
def _blend_single(a, b, mode, weight):
    ax, ay, az = _validate_normal(a, "a")
    bx, by, bz = _validate_normal(b, "b")

    if weight is None:
        if mode == "udn":
            x, y, z = ax + bx, ay + by, az
        else:  # whiteout
            x, y, z = ax + bx, ay + by, az * bz
        return _normalize_xyz(x, y, z, what="blended normal")

    # 权重混合：线性插值后重新归一；w=0/1 精确退化为单个输入。
    if weight == 0.0:
        return _unit_or_normalized((ax, ay, az), what="normal a")
    if weight == 1.0:
        return _unit_or_normalized((bx, by, bz), what="normal b")
    x = ax + weight * (bx - ax)
    y = ay + weight * (by - ay)
    z = az + weight * (bz - az)
    return _normalize_xyz(x, y, z, what="blended normal")


def _tbn_single(n, t):
    nx, ny, nz = _validate_normal(n, "n")
    nx, ny, nz = _unit_or_normalized((nx, ny, nz), what="normal n")

    if t is not None:
        try:
            tx, ty, tz = t
        except (TypeError, ValueError):
            raise InvalidTangentError(
                "tangent must be a sequence of 3 numbers, got %r" % (_fmt_error_value(t),)
            ) from None
        if (
            isinstance(tx, (tuple, list))
            or isinstance(ty, (tuple, list))
            or isinstance(tz, (tuple, list))
        ):
            raise InvalidTangentError("tangent must be a sequence of 3 numbers")
        try:
            tx = _FLOAT(tx)
            ty = _FLOAT(ty)
            tz = _FLOAT(tz)
        except (TypeError, ValueError):
            raise InvalidTangentError(
                "tangent must contain real numbers, got %r" % (_fmt_error_value(t),)
            ) from None
        for c in (tx, ty, tz):
            if _ISNAN(c):
                raise NaNTangentError(
                    "tangent contains a NaN component: %r" % (_fmt_error_value(t),)
                )
        for c in (tx, ty, tz):
            if not _ISFINITE(c):
                raise InfiniteTangentError(
                    "tangent contains an infinite component: %r" % (_fmt_error_value(t),)
                )
        d2t = tx * tx + ty * ty + tz * tz
        if d2t == 0.0:
            raise InvalidTangentError("tangent is the zero vector")
        # Gram-Schmidt：去掉沿法线方向的分量。
        dot = tx * nx + ty * ny + tz * nz
        tx -= dot * nx
        ty -= dot * ny
        tz -= dot * nz
        d2t = tx * tx + ty * ty + tz * tz
        if not (0.0 < d2t < _INF):
            raise InvalidTangentError("tangent is parallel to the normal")
        s = 1.0 / _SQRT(d2t)
        tx, ty, tz = tx * s, ty * s, tz * s
    else:
        # 未提供切线：选择与法线夹角最小的坐标轴作为种子，构造稳定切线。
        ax, ay, az = abs(nx), abs(ny), abs(nz)
        if ax <= ay and ax <= az:
            tx, ty, tz = 1.0, 0.0, 0.0
        elif ay <= az:
            tx, ty, tz = 0.0, 1.0, 0.0
        else:
            tx, ty, tz = 0.0, 0.0, 1.0
        dot = tx * nx + ty * ny + tz * nz
        tx -= dot * nx
        ty -= dot * ny
        tz -= dot * nz
        d2t = tx * tx + ty * ty + tz * tz
        s = 1.0 / _SQRT(d2t)
        tx, ty, tz = tx * s, ty * s, tz * s

    # 右手基：B = N × T。
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    return (tx, ty, tz), (bx, by, bz), (nx, ny, nz)


# ---------------------------------------------------------------------------
# 批量核心（NumPy）
# ---------------------------------------------------------------------------
def _check_array(arr, name):
    if arr.shape[-1] != 3:
        raise NormalDimensionError(
            "%s must have shape (..., 3), got %r" % (name, (arr.shape,))
        )
    if bool(_np.isnan(arr).any()):
        raise NaNComponentError("%s contains a NaN component" % name)
    if not bool(_np.isfinite(arr).all()):
        raise InfiniteComponentError("%s contains an infinite component" % name)
    if not bool(_np.any(arr != 0.0, axis=-1).all()):
        raise ZeroNormalError("%s contains a zero vector" % name)


def _normalize_array(arr, *, error=DegenerateBlendError, what="blended normal"):
    """按最后一维归一化，先按行最大绝对值缩放以避免溢出。"""
    m = _np.max(_np.abs(arr), axis=-1, keepdims=True)
    bad = m == 0.0
    if bool(bad.any()):
        idx = _np.argwhere(bad.reshape(-1))[0]
        raise error("%s is degenerate at flat index %d" % (what, int(idx[0])))
    scaled = arr / m
    d2 = _np.sum(scaled * scaled, axis=-1, keepdims=True)
    if not bool(_np.isfinite(d2).all()):
        raise error("%s is degenerate (non-finite length)" % what)
    return scaled / _np.sqrt(d2)


def _as_array(v, name):
    try:
        arr = _np.asarray(v, dtype=_np.float64)
    except (TypeError, ValueError):
        raise NormalDimensionError(
            "%s must be an array of shape (..., 3)" % name
        ) from None
    if arr.ndim == 0:
        raise NormalDimensionError("%s must have at least one dimension" % name)
    if arr.dtype != _np.float64:
        try:
            arr = arr.astype(_np.float64, copy=False)
        except (TypeError, ValueError):
            raise NormalDimensionError(
                "%s must contain real numbers" % name
            ) from None
    return arr


def _weight_array(weight, shape):
    if weight is None:
        return None
    try:
        arr = _np.asarray(weight, dtype=_np.float64)
    except (TypeError, ValueError):
        raise InvalidWeightError("weight must contain real numbers") from None
    if not bool(_np.isfinite(arr).all()):
        raise InvalidWeightError("weight must be finite")
    if not bool(((arr >= 0.0) & (arr <= 1.0)).all()):
        raise InvalidWeightError("weight must be in [0, 1]")
    try:
        arr = _np.broadcast_to(arr, shape)
        return arr
    except ValueError:
        pass
    # 允许逐像素权重（形状为法线形状去掉最后一维）。
    try:
        arr = _np.broadcast_to(arr, shape[:-1])
    except ValueError:
        raise InvalidWeightError(
            "weight shape %r cannot broadcast to %r" % (arr.shape, shape)
        ) from None
    return _np.repeat(arr[..., _np.newaxis], shape[-1], axis=-1)


def _blend_arrays(a, b, mode, weight):
    A = _as_array(a, "a")
    B = _as_array(b, "b")
    if A.ndim < 1 or B.ndim < 1:
        raise NormalDimensionError("a and b must be arrays of normals")
    _check_array(A, "a")
    _check_array(B, "b")

    try:  # 单个法线可广播到整幅贴图。
        A, B = _np.broadcast_arrays(A, B)
    except ValueError:
        raise NormalDimensionError(
            "a shape %r and b shape %r are not broadcast-compatible" % (A.shape, B.shape)
        ) from None
    if A.shape[-1] != 3:
        raise NormalDimensionError("last dimension must be 3")

    shape = A.shape
    if weight is None:
        if mode == "udn":
            out = _np.empty(shape, dtype=_np.float64)
            out[..., 0] = A[..., 0] + B[..., 0]
            out[..., 1] = A[..., 1] + B[..., 1]
            out[..., 2] = A[..., 2]
        else:
            out = _np.empty(shape, dtype=_np.float64)
            out[..., 0] = A[..., 0] + B[..., 0]
            out[..., 1] = A[..., 1] + B[..., 1]
            out[..., 2] = A[..., 2] * B[..., 2]
        return _normalize_array(out, what="blended normal")

    w = _weight_array(weight, shape)
    # 全 0 / 全 1 权重严格退化为单输入（仍归一化以满足“输出必为单位法线”）。
    if bool((w == 0.0).all()):
        return _normalize_array(A, what="normal a")
    if bool((w == 1.0).all()):
        return _normalize_array(B, what="normal b")
    out = A + w * (B - A)
    return _normalize_array(out, what="blended normal")


def _tuplify(arr):
    """把结果数组转成嵌套的 3 元组结构（保持前导维度）。"""
    if arr.ndim == 2:
        return [tuple(row) for row in arr.tolist()]
    if arr.ndim < 2:
        return tuple(arr.tolist())
    return [_tuplify(sub) for sub in arr]


def _tbn_arrays(n, t):
    """向量化版本的 :func:`_tbn_single`；支持 ``(..., 3)`` 批量输入。"""
    N = _as_array(n, "n")
    if N.shape[-1] != 3:
        raise NormalDimensionError("n must have shape (..., 3), got %r" % (N.shape,))
    _check_array(N, "n")
    N = _normalize_array(N, what="normal n")
    if t is None:
        # 选与法线最不平行的坐标轴作种子（逐元素）。
        shape = N.shape[:-1]
        seed = _np.zeros(shape + (3,), dtype=_np.float64)
        absN = _np.abs(N)
        axis = _np.argmin(absN, axis=-1)
        flat = seed.reshape(-1, 3)
        flat[_np.arange(flat.shape[0]), axis.reshape(-1)] = 1.0
        T = seed.reshape(shape + (3,))
    else:
        T = _as_array(t, "t")
        if T.shape[-1] != 3:
            raise NormalDimensionError("t must have shape (..., 3), got %r" % (T.shape,))
        if bool(_np.isnan(T).any()):
            raise NaNTangentError("tangent contains a NaN component")
        if not bool(_np.isfinite(T).all()):
            raise InfiniteTangentError("tangent contains an infinite component")
        T = _np.broadcast_to(T, N.shape)
        if bool((_np.abs(T).sum(axis=-1) == 0.0).any()):
            raise InvalidTangentError("tangent is the zero vector")
    dot = _np.sum(T * N, axis=-1, keepdims=True)
    T = T - dot * N
    # 与法线平行时 Gram-Schmidt 结果为零向量：_normalize_array 抛出
    # InvalidTangentError（可区分的异常分类）。
    T = _normalize_array(T, error=InvalidTangentError, what="tangent")
    B = _np.cross(N, T)
    return T, B, N


# ---------------------------------------------------------------------------
# 批量核心（纯 Python，NumPy 缺失或未启用时的退路）
# ---------------------------------------------------------------------------
def _leading_shape(v, name):
    """返回 ``(shape, leaves)``：``shape`` 是前导维度，``leaves`` 是法线三元组。"""
    if _is_single(v):
        x, y, z = _validate_normal(v, name)
        return (), [(x, y, z)]
    try:
        items = list(v)
    except TypeError:
        raise NormalDimensionError(
            "%s must be a normal or a nested sequence of normals, got %r"
            % (name, _fmt_error_value(v))
        ) from None
    if not items:
        raise NormalDimensionError("%s is empty" % name)
    child_shape = None
    leaves = []
    for child in items:
        s, ls = _leading_shape(child, name)
        if child_shape is None:
            child_shape = s
        elif s != child_shape:
            raise NormalDimensionError("%s is ragged (mixes %r with %r)" % (name, child_shape, s))
        leaves.extend(ls)
    return (len(items),) + child_shape, leaves


def _weight_leaves(weight, count):
    """把权重展开成每个像素一个标量（``None`` 表示不做权重混合）。"""
    if weight is None:
        return [None] * count
    if _is_number(weight) and not isinstance(weight, (list, tuple)):
        w = _parse_weight(weight)
        return [w] * count
    try:
        items = list(weight)
    except TypeError:
        raise InvalidWeightError(
            "weight must be a scalar or a nested sequence, got %r"
            % (_fmt_error_value(weight),)
        ) from None
    flat = []
    stack = [iter(items)]
    while stack:
        try:
            node = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(node, (list, tuple)):
            stack.append(iter(node))
        else:
            flat.append(_parse_weight(node))
    if len(flat) == 1:
        return flat * count
    if len(flat) != count:
        raise InvalidWeightError(
            "weight has %d values but there are %d normals" % (len(flat), count)
        )
    return flat


def _rebuild(shape, values):
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    step = len(values) // shape[0]
    return [_rebuild(shape[1:], values[i * step:(i + 1) * step]) for i in range(shape[0])]


def _blend_python(a, b, mode, weight):
    sa, la = _leading_shape(a, "a")
    sb, lb = _leading_shape(b, "b")
    if sa == ():
        shape = sb
        pairs = [(la[0], y) for y in lb]
    elif sb == ():
        shape = sa
        pairs = [(x, lb[0]) for x in la]
    elif sa == sb:
        shape = sa
        pairs = list(zip(la, lb))
    else:
        raise NormalDimensionError(
            "a shape %r and b shape %r are not broadcast-compatible" % (sa, sb)
        )
    weights = _weight_leaves(weight, len(pairs))
    results = _blend_pairs_fast(pairs, mode, weights)
    return _rebuild(shape, results)


def _blend_pairs_fast(pairs, mode, weights):
    """批量算术的纯 Python 快速路径（输入已通过 :func:`_validate_normal`）。

    与 ``_blend_single`` 使用同一公式；这里内联循环以摊薄函数调用开销。
    """
    out = []
    append = out.append
    sqrt = _SQRT
    isfinite = _ISFINITE
    inf = _INF
    whiteout = mode == "whiteout"
    for (a, b), w in zip(pairs, weights):
        ax, ay, az = a
        bx, by, bz = b
        if w is None:
            x = ax + bx
            y = ay + by
            z = az * bz if whiteout else az
        elif w == 0.0:
            append(_unit_or_normalized(a, what="normal a"))
            continue
        elif w == 1.0:
            append(_unit_or_normalized(b, what="normal b"))
            continue
        else:
            x = ax + w * (bx - ax)
            y = ay + w * (by - ay)
            z = az + w * (bz - az)
        d2 = x * x + y * y + z * z
        if 0.0 < d2 < inf and isfinite(d2):
            s = 1.0 / sqrt(d2)
            append((x * s, y * s, z * s))
            continue
        # 零向量 / 溢出 / 下溢：交给稳健路径（它会抛出可区分的异常）。
        append(_normalize_xyz(x, y, z, what="blended normal"))
    return out


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
class NormalBlend:
    """法线贴图混合器。

    实例无共享可变状态；所有方法都是纯函数，可安全地在多个线程中并发调用。
    """

    __slots__ = ("_default_mode", "_default_weight")

    def __init__(self, weight=None, mode="udn"):
        # 兼容 ``NormalBlend('whiteout')`` 这种把模式放在首位的写法。
        if isinstance(weight, str) and isinstance(mode, str):
            weight, mode = None, weight
        self._default_mode = _parse_mode(mode)
        self._default_weight = None if weight is None else self._store_weight(weight)

    @staticmethod
    def _store_weight(weight):
        if _np is not None and isinstance(weight, _np.ndarray) and weight.ndim > 0:
            return weight.astype(_np.float64, copy=True)
        if isinstance(weight, (list, tuple)):
            return _np.asarray(weight, dtype=_np.float64) if _np is not None else tuple(weight)
        return _parse_weight(weight)

    def tbn(self, n, t=None):
        """由法线 ``n`` 与切线 ``t`` 构造正交切线基 ``(T, B, N)``。

        * 归一化 ``n``；
        * 对 ``t`` 做 Gram-Schmidt，得到与 ``n`` 正交的单位切线；
        * ``B = N × T``，保证 ``(T, B, N)`` 是右手正交基；
        * ``t`` 省略时自动选取一个稳定切线；``t`` 非法（零向量/与 ``n`` 共线）
          抛 :class:`InvalidTangentError`。

        ``n`` 也可以是 ``(..., 3)`` 的数组（批量法线），此时返回三个同形状
        的 ndarray。
        """
        if _np is not None:
            n_batch = isinstance(n, _np.ndarray) and n.ndim > 1
            t_batch = isinstance(t, _np.ndarray) and t.ndim > 1
            if n_batch or t_batch:
                return _tbn_arrays(n, t)
        return _tbn_single(n, t)

    # 便捷变换：切线空间 ↔ 世界空间，保持单位长度与正交性。
    def transform(self, v, n, t=None):
        """把切线空间向量 ``v`` 变换到由 ``tbn(n, t)`` 定义的世界空间。"""
        T, B, N = self.tbn(n, t)
        vx, vy, vz = _validate_normal(v, "v")
        return (
            vx * T[0] + vy * B[0] + vz * N[0],
            vx * T[1] + vy * B[1] + vz * N[1],
            vx * T[2] + vy * B[2] + vz * N[2],
        )

    def to_world(self, v, n, t=None):
        return self.transform(v, n, t)

    def to_tangent(self, v, n, t=None):
        """把世界空间向量 ``v`` 变换回切线空间（转置即为逆）。"""
        T, B, N = self.tbn(n, t)
        vx, vy, vz = _validate_normal(v, "v")
        return (
            vx * T[0] + vy * T[1] + vz * T[2],
            vx * B[0] + vy * B[1] + vz * B[2],
            vx * N[0] + vy * N[1] + vz * N[2],
        )

    def blend(self, a, b, *args, **kwargs):
        """混合两个法线（或两幅法线贴图）。

        参数
        ----
        a, b : 3 元组 / 长度为 3 的序列，或形如 ``(..., 3)`` 的数组/嵌套序列。
        mode : ``'udn'``（默认，partial-derivative）或 ``'whiteout'``。
        weight : ``None`` 表示直接叠加；``[0, 1]`` 内实数表示从 ``a`` 到 ``b``
            的线性权重（0/1 精确退化为单输入），也可传入逐像素权重数组。

        返回
        ----
        单个法线返回 ``(x, y, z)`` 三元组；批量/图像输入返回结构相同的
        嵌套三元组列表（NumPy 输入返回 ``float64`` ndarray）。结果已重归一。
        """
        mode_arg = None
        weight_arg = None
        if "mode" in kwargs:
            mode_arg = kwargs.pop("mode")
        for key in _WEIGHT_ALIASES:
            if key in kwargs:
                if weight_arg is not None:
                    raise TypeError("multiple weight arguments given")
                weight_arg = kwargs.pop(key)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError("blend() got an unexpected keyword argument %r" % unexpected)
        for arg in args:
            if isinstance(arg, str):
                if mode_arg is not None:
                    raise TypeError("blend() got multiple values for 'mode'")
                mode_arg = arg
            else:
                if weight_arg is not None:
                    raise TypeError("blend() got multiple values for 'weight'")
                weight_arg = arg

        mode = self._default_mode if mode_arg is None else _parse_mode(mode_arg)

        if weight_arg is None:
            weight = self._default_weight
        else:
            weight = self._store_weight(weight_arg)
        if _np is not None and isinstance(weight, _np.ndarray):
            scalar_weight = float(weight.reshape(-1)[0]) if weight.size == 1 else None
        else:
            scalar_weight = weight
        if scalar_weight is not None:
            weight = _parse_weight(scalar_weight)

        a_single = _is_single(a)
        b_single = _is_single(b)
        if a_single and b_single:
            if weight is not None and not isinstance(weight, float):
                raise InvalidWeightError("weight must be a scalar for a single normal")
            return _blend_single(a, b, mode, weight)
        if _np is None:
            return _blend_python(a, b, mode, weight)
        if a_single:
            a = self._as_single_array(a, "a")
        elif b_single:
            b = self._as_single_array(b, "b")

        out = _blend_arrays(a, b, mode, weight)
        if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
            return out
        return _tuplify(out)

    @staticmethod
    def _as_single_array(v, name):
        if _np is None:
            # 无 NumPy：把单个法线展开成与批量输入相同的所有位置。
            return v
        arr = _np.asarray(v, dtype=_np.float64)
        if arr.shape != (3,):
            raise NormalDimensionError("%s must be a single normal" % name)
        return arr

    # ---- 批量便捷入口（等同于传数组给 blend） ----
    def blend_normals(self, a, b, *args, **kwargs):
        return self.blend(a, b, *args, **kwargs)

    blend_batch = blend_normals
    blend_many = blend_normals
    blend_pixels = blend_normals
    blend_image = blend_normals


# ---------------------------------------------------------------------------
# 模块级便捷函数（无状态）
# ---------------------------------------------------------------------------
_DEFAULT = NormalBlend()


def blend_normals(a, b, *args, **kwargs):
    """模块级快捷方式：``NormalBlend().blend(a, b, ...)``。"""
    return _DEFAULT.blend(a, b, *args, **kwargs)


def tbn(n, t=None):
    """模块级快捷方式：``NormalBlend().tbn(n, t)``。"""
    return _DEFAULT.tbn(n, t)
