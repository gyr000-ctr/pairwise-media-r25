"""旋转体网格生成。

将 (r, y) 平面上的闭合轮廓绕给定轴旋转一周，生成三角网格：

- 顶点按 ``环主序`` 排列：``index = ring * len(profile) + i``，
  接缝处不复制顶点（第 ``segments`` 环回绕到第 0 环），天然焊接。
- 法线用解析式：轮廓切线 T 与旋转切向 dP/dθ 的叉积方向，
  即 ``n = sign * (dy*cosθ, dy*sinθ, -dr)``（在旋转正交基下），单位化。
- 体积与 Pappus 定理一致（网格外法线朝外、绕序一致）。
- 凹轮廓允许；自交轮廓抛出 :class:`SelfIntersectionError`。
- 错误分类：非有限输入、退化轮廓、非法轴分别使用不同的异常子类，
  全部继承自 ``ValueError``。

线程安全：``build`` 的全部中间状态都在栈上，仅在返回前把法线
一次性写入 ``self._last_normals``（单个原子赋值），并发调用
``build`` 不会产生损坏状态；不同实例之间完全独立。
"""

import math

__all__ = [
    "Revolve",
    "RevolveError",
    "NonFiniteInputError",
    "DegenerateProfileError",
    "SelfIntersectionError",
    "InvalidAxisError",
]

_TWO_PI = 2.0 * math.pi


class RevolveError(ValueError):
    """旋转体构建错误基类（继承 ValueError，可与普通 ValueError 区分）。"""


class NonFiniteInputError(RevolveError):
    """输入包含 NaN 或无穷。"""


class DegenerateProfileError(RevolveError):
    """轮廓退化（空、点数不足、零面积、零长度切线等）。"""


class SelfIntersectionError(RevolveError):
    """轮廓自交。"""


class InvalidAxisError(RevolveError):
    """旋转轴非法（分量数不对、非数值、零向量等）。"""


def _check_number(value, exc_struct, what):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise exc_struct("%s 必须是实数, 得到 %r" % (what, value))
    if not math.isfinite(value):
        raise NonFiniteInputError("%s 包含非有限值: %r" % (what, value))
    return float(value)


def _validate_profile(profile):
    if isinstance(profile, (str, bytes)):
        raise DegenerateProfileError("profile 必须是 (r, y) 点的可迭代序列")
    try:
        raw = list(profile)
    except TypeError:
        raise DegenerateProfileError("profile 必须是 (r, y) 点的可迭代序列") from None
    pts = []
    for p in raw:
        try:
            r, y = p
        except (TypeError, ValueError):
            raise DegenerateProfileError(
                "profile 的每个点必须是 (r, y) 二元组, 得到 %r" % (p,)) from None
        pts.append((_check_number(r, DegenerateProfileError, "profile 半径"),
                    _check_number(y, DegenerateProfileError, "profile 高度")))
    if len(pts) < 3:
        raise DegenerateProfileError(
            "轮廓退化: 至少需要 3 个点, 得到 %d 个" % len(pts))
    return pts


def _validate_segments(segments):
    if isinstance(segments, bool) or not isinstance(segments, int):
        raise RevolveError("segments 必须是整数, 得到 %r" % (segments,))
    if segments < 3:
        raise RevolveError("segments 必须 >= 3, 得到 %d" % segments)
    return segments


def _basis(axis):
    try:
        ax, ay, az = axis
    except (TypeError, ValueError):
        raise InvalidAxisError("axis 必须是三分量向量, 得到 %r" % (axis,)) from None
    ax = _check_number(ax, InvalidAxisError, "axis 分量")
    ay = _check_number(ay, InvalidAxisError, "axis 分量")
    az = _check_number(az, InvalidAxisError, "axis 分量")
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length == 0.0:
        raise InvalidAxisError("axis 不能是零向量")
    wx, wy, wz = ax / length, ay / length, az / length
    # 选与 w 最不平行的坐标轴作为参考，构造右手正交基 (u, v, w)
    if abs(wx) <= abs(wy) and abs(wx) <= abs(wz):
        rx, ry, rz = 1.0, 0.0, 0.0
    elif abs(wy) <= abs(wz):
        rx, ry, rz = 0.0, 1.0, 0.0
    else:
        rx, ry, rz = 0.0, 0.0, 1.0
    # u = normalize(r × w)
    ux, uy, uz = ry * wz - rz * wy, rz * wx - rx * wz, rx * wy - ry * wx
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    # v = w × u，于是 u × v = w（右手系）
    vx, vy, vz = wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux
    return (ux, uy, uz), (vx, vy, vz), (wx, wy, wz)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(p, q, x, eps):
    return (min(p[0], q[0]) - eps <= x[0] <= max(p[0], q[0]) + eps and
            min(p[1], q[1]) - eps <= x[1] <= max(p[1], q[1]) + eps)


def _segments_intersect(p1, p2, p3, p4, eps):
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if (((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and
            ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))):
        return True
    if abs(d1) <= eps and _on_segment(p3, p4, p1, eps):
        return True
    if abs(d2) <= eps and _on_segment(p3, p4, p2, eps):
        return True
    if abs(d3) <= eps and _on_segment(p1, p2, p3, eps):
        return True
    if abs(d4) <= eps and _on_segment(p1, p2, p4, eps):
        return True
    return False


def _check_self_intersection(pts):
    """闭合折线的自交检测（允许凹轮廓，不允许任何边相交/相触）。"""
    n = len(pts)
    boxes = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        boxes.append((min(a[0], b[0]), min(a[1], b[1]),
                      max(a[0], b[0]), max(a[1], b[1])))
    extent = max(max(b[2] for b in boxes) - min(b[0] for b in boxes),
                 max(b[3] for b in boxes) - min(b[1] for b in boxes))
    eps = 1e-12 * extent
    for i in range(n):
        bi = boxes[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # 闭合处相邻边
            bj = boxes[j]
            if (bi[0] > bj[2] + eps or bj[0] > bi[2] + eps or
                    bi[1] > bj[3] + eps or bj[1] > bi[3] + eps):
                continue
            p3 = pts[j]
            p4 = pts[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4, eps):
                raise SelfIntersectionError(
                    "轮廓自交: 边 %d 与边 %d 相交" % (i, j))


class Revolve:
    """旋转体网格生成器。"""

    def __init__(self):
        self._last_normals = None

    def build(self, profile, segments, axis=(0.0, 1.0, 0.0)):
        """生成旋转体网格。

        :param profile: (r, y) 点序列，按闭合轮廓处理（末点连回首点）。
        :param segments: 旋转分段数（>= 3）。
        :param axis: 过原点的旋转轴（三分量非零向量）。
        :returns: ``(vertices, triangles, normals)`` 三元组；
            顶点 ``ring * len(profile) + i``，三角形绕序一致且法线朝外。
        """
        pts = _validate_profile(profile)
        seg = _validate_segments(segments)
        (ux, uy, uz), (vx, vy, vz), (wx, wy, wz) = _basis(axis)
        _check_self_intersection(pts)
        n = len(pts)

        rs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # 轮廓有向面积 ×2，判断绕向并排除零面积退化轮廓
        area2 = 0.0
        for i in range(n):
            area2 += rs[i] * ys[(i + 1) % n] - rs[(i + 1) % n] * ys[i]
        extent = max(max(rs) - min(rs), max(ys) - min(ys))
        if abs(area2) <= 1e-12 * extent * extent:
            raise DegenerateProfileError("轮廓退化: 面积为零（共线或重复点）")
        ccw = area2 > 0.0
        sign = 1.0 if ccw else -1.0

        # 轮廓切线（中心差分）→ 解析法线系数：
        # n(θ) = sign * (dy/L * d(θ) + (-dr/L) * w)，d(θ) = cosθ·u + sinθ·v
        na = []
        bw = []
        for i in range(n):
            dr = rs[(i + 1) % n] - rs[i - 1]
            dy = ys[(i + 1) % n] - ys[i - 1]
            ln = math.hypot(dr, dy)
            if ln == 0.0:
                raise DegenerateProfileError(
                    "轮廓退化: 点 %d 处切线为零（折返点）" % i)
            s = sign / ln
            na.append(dy * s)
            b = -dr * s
            bw.append((b * wx, b * wy, b * wz))

        # 每个轮廓点的轴向偏移 y·w
        yw = [(y * wx, y * wy, y * wz) for y in ys]

        # 每环的径向单位方向 d(θ) = cosθ·u + sinθ·v
        rings = []
        for k in range(seg):
            t = _TWO_PI * k / seg
            c = math.cos(t)
            sn = math.sin(t)
            rings.append((c * ux + sn * vx, c * uy + sn * vy, c * uz + sn * vz))

        verts = []
        norms = []
        vextend = verts.extend
        nextend = norms.extend
        zrs = list(zip(rs, yw))
        zna = list(zip(na, bw))
        for k in range(seg):
            dx, dy_, dz = rings[k]
            vextend([(r * dx + ywx, r * dy_ + ywy, r * dz + ywz)
                     for r, (ywx, ywy, ywz) in zrs])
            nextend([(a * dx + bwx, a * dy_ + bwy, a * dz + bwz)
                     for a, (bwx, bwy, bwz) in zna])

        # 三角形：接缝处回绕到第 0 环（顶点焊接），绕序与法线一致朝外
        ip1 = list(range(1, n)) + [0]
        tris = []
        tappend = tris.append
        for k in range(seg):
            k0 = k * n
            k1 = ((k + 1) % seg) * n
            for i in range(n):
                j = ip1[i]
                a = k0 + i
                b = k0 + j
                c = k1 + j
                d = k1 + i
                if ccw:
                    tappend((a, c, b))
                    tappend((a, d, c))
                else:
                    tappend((a, b, c))
                    tappend((a, c, d))

        self._last_normals = norms
        return (verts, tris, norms)

    def normals(self):
        """返回上一次 :meth:`build` 生成的单位法线列表。"""
        if self._last_normals is None:
            raise RevolveError("normals() 之前需要先调用 build()")
        return self._last_normals
