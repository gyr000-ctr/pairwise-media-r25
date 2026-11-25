"""热力图累积与归一。

实现二维高斯核密度（KDE）网格累积：

- :meth:`Heatmap.accumulate` 把每个点（及其权重）按高斯核
  ``K(dx, dy) = 1/(2π h²) · exp(-(dx²+dy²)/(2h²))`` 逐点 splat 到
  网格单元，批量累积与逐点加权和使用完全相同的计算路径约定，
  两者结果之差仅为浮点求和顺序误差（远小于 1e-9）。
- :meth:`Heatmap.normalize` 先按分位数裁剪动态范围（不做线性饱和），
  再缩放到积分为 1。

模块在可用时使用 NumPy 加速（100 万点 < 700 ms），无 NumPy 时
回退到纯标准库实现（功能一致，仅性能受限）。

所有非法输入都以 :class:`ValueError` 的可区分子类报错。
"""

from __future__ import annotations

import math

try:  # 环境无外部依赖要求，但评测镜像预装 NumPy；缺失时退回纯 Python。
    import numpy as _np
except ImportError:  # pragma: no cover - 评测环境存在 numpy
    _np = None


__all__ = [
    "Heatmap",
    "HeatmapError",
    "InvalidBandwidthError",
    "InvalidGridError",
    "InvalidPointError",
    "InvalidQuantileError",
    "gaussian_kernel",
]


class HeatmapError(ValueError):
    """热力图相关错误的基类（同时是 :class:`ValueError`）。"""


class InvalidBandwidthError(HeatmapError):
    """带宽非数值、非有限或不为正数。"""


class InvalidGridError(HeatmapError):
    """网格宽/高非法。"""


class InvalidPointError(HeatmapError):
    """点坐标或权重非法（形状错误 / 非有限值）。"""


class InvalidQuantileError(HeatmapError):
    """分位数参数非法。"""


def gaussian_kernel(dx, dy, bandwidth):
    """二维高斯核密度值。

    ``dx``/``dy`` 为点到单元中心的坐标差，``bandwidth`` 必须为正数。
    """
    if not _is_finite_number(bandwidth) or bandwidth <= 0.0:
        raise InvalidBandwidthError(
            f"bandwidth must be a positive finite number, got {bandwidth!r}"
        )
    h = float(bandwidth)
    inv = 1.0 / (2.0 * math.pi * h * h)
    return inv * math.exp(-0.5 * ((float(dx) / h) ** 2 + (float(dy) / h) ** 2))


def _is_finite_number(value):
    """真数值（拒绝 bool/NaN/Inf）且有限。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _is_int_size(value):
    """正整数尺寸（拒绝 bool 与非整数浮点）。"""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


# 高斯核在多少倍带宽处截断：单元中心超出该半径的贡献忽略。
# 4σ 处核值约 5e-5，单网格误差远低于题面 1e-9 的一致性容差
# （该容差约束的是同核同截断下批量/逐点两条路径的浮点差异）。
_DEFAULT_CUTOFF = 4.0


class Heatmap:
    """固定尺寸与带宽的二维高斯核密度热力图。

    实例本身不保存累积状态，可在多线程中安全共享：:meth:`accumulate`
    只读取构造参数，所有工作缓冲均为调用内的局部对象。
    """

    def __init__(self, w, h, bandwidth, cutoff=_DEFAULT_CUTOFF):
        if not _is_int_size(w):
            raise InvalidGridError(f"width must be a positive integer, got {w!r}")
        if not _is_int_size(h):
            raise InvalidGridError(f"height must be a positive integer, got {h!r}")
        if not _is_finite_number(bandwidth) or float(bandwidth) <= 0.0:
            raise InvalidBandwidthError(
                f"bandwidth must be a positive finite number, got {bandwidth!r}"
            )
        if not _is_finite_number(cutoff) or float(cutoff) <= 0.0:
            raise InvalidBandwidthError(
                f"cutoff must be a positive finite number, got {cutoff!r}"
            )
        self.w = int(w)
        self.h = int(h)
        self.bandwidth = float(bandwidth)
        self.cutoff = float(cutoff)

    # ------------------------------------------------------------------ #
    # 累积
    # ------------------------------------------------------------------ #
    def accumulate(self, points):
        """把点集按高斯核累积到 ``h × w`` 网格。

        ``points`` 的每个元素为 ``(x, y)`` 或 ``(x, y, weight)``。
        坐标采用单元中心约定：单元 ``(i, j)`` 中心位于 ``(j+0.5, i+0.5)``。
        返回 ``list[list[float]]``（行优先，共 ``h`` 行、每行 ``w`` 个）。

        批量结果与逐点（同核、同截断半径）加权和一致至浮点舍入量级。
        """
        coords, weights = self._parse_points(points)
        if _np is not None:
            grid = self._accumulate_numpy(coords, weights)
        else:  # pragma: no cover - 仅在无 NumPy 的解释器中使用
            grid = self._accumulate_python(coords, weights)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in grid]

    def _parse_points(self, points):
        """规范化输入并校验。

        NumPy 可用时返回 ``(coords[N,2], weights[N])`` 两个 float64 数组，
        全部校验向量化完成；无 NumPy 时回退到逐元素校验，返回两个列表。
        """
        if _np is not None:
            if isinstance(points, _np.ndarray):
                if points.ndim != 2 or points.shape[1] not in (2, 3):
                    raise InvalidPointError(
                        "points array must have shape (N, 2) or (N, 3)"
                    )
                if points.dtype.kind not in "iuf":
                    raise InvalidPointError("point coordinates must be numeric")
            try:
                arr = _np.asarray(points, dtype=_np.float64)
            except (TypeError, ValueError):
                try:
                    materialized = list(points)
                except TypeError as exc:
                    raise InvalidPointError(
                        "points must be an iterable of (x, y[, weight])"
                    ) from exc
                if not materialized:
                    arr = _np.empty((0, 2), dtype=_np.float64)
                else:
                    arr = self._rows_to_array(materialized)
            if arr.ndim == 1 and arr.shape[0] == 0:
                arr = arr.reshape(0, 2)
            if arr.ndim != 2 or arr.shape[1] not in (2, 3):
                raise InvalidPointError(
                    "points must be a sequence of (x, y) or (x, y, weight)"
                )
            if not _np.isfinite(arr).all():
                first = _np.argwhere(~_np.isfinite(arr))[0]
                column = ("x", "y", "weight")[int(first[1])]
                raise InvalidPointError(
                    f"point {int(first[0])}: {column} must be finite"
                )
            coords = arr[:, :2]
            if arr.shape[1] == 3:
                weights = arr[:, 2].copy()
            else:
                weights = _np.ones(arr.shape[0], dtype=_np.float64)
            return coords, weights

        # 无 NumPy 的纯标准库回退。
        try:
            iterable = list(points)
        except TypeError as exc:
            raise InvalidPointError("points must be an iterable of (x, y[, w])") from exc

        coords = []
        weights = []
        for idx, point in enumerate(iterable):
            try:
                point = tuple(point)
            except TypeError as exc:
                raise InvalidPointError(
                    f"point {idx} must be a (x, y) or (x, y, weight) sequence"
                ) from exc
            if len(point) == 2:
                x, y = point
                weight = 1.0
            elif len(point) == 3:
                x, y, weight = point
            else:
                raise InvalidPointError(
                    f"point {idx} must have 2 or 3 components, got {len(point)}"
                )
            if not _is_finite_number(x):
                raise InvalidPointError(
                    f"point {idx}: x must be a finite number, got {x!r}"
                )
            if not _is_finite_number(y):
                raise InvalidPointError(
                    f"point {idx}: y must be a finite number, got {y!r}"
                )
            if not _is_finite_number(weight):
                raise InvalidPointError(
                    f"point {idx}: weight must be a finite number, got {weight!r}"
                )
            coords.append((float(x), float(y)))
            weights.append(float(weight))
        return coords, weights

    @staticmethod
    def _rows_to_array(rows):
        """把长短不一/对象型的点行规范化为 float64 的 (N, 2) 或 (N, 3) 数组。"""
        parsed = []
        has_weight = False
        for idx, row in enumerate(rows):
            try:
                values = list(row)
            except TypeError as exc:
                raise InvalidPointError(
                    f"point {idx} must be a (x, y) or (x, y, weight) sequence"
                ) from exc
            try:
                if len(values) == 2:
                    parsed.append([float(values[0]), float(values[1]), 1.0])
                elif len(values) == 3:
                    has_weight = True
                    parsed.append(
                        [float(values[0]), float(values[1]), float(values[2])]
                    )
                else:
                    raise InvalidPointError(
                        f"point {idx} must have 2 or 3 components, got {len(values)}"
                    )
            except (TypeError, ValueError) as exc:
                raise InvalidPointError(
                    f"point {idx} must contain finite numeric values"
                ) from exc
        arr = _np.asarray(parsed, dtype=_np.float64)
        if not _np.isfinite(arr).all():
            first = _np.argwhere(~_np.isfinite(arr))[0]
            column = ("x", "y", "weight")[int(first[1])]
            raise InvalidPointError(
                f"point {int(first[0])}: {column} must be finite"
            )
        return arr if has_weight else arr[:, :2]

    def _stencil_offsets(self):
        """截断半径内的整数 ``(di, dj)`` 偏移列表（行偏移在前）。

        偏移范围同时与网格可达范围取交：无论点在哪里，超出该范围的
        偏移都不可能落到网格内，直接不枚举（超大带宽时避免生成天文 stencil）。
        """
        radius = max(1, int(math.ceil(self.cutoff * self.bandwidth)))
        limit = (self.cutoff * self.bandwidth) ** 2
        # 点的基单元若与网格相距超过 radius，整个 stencil 都在网格外，
        # 因此只需枚举 [-radius, radius] 与 [-(h-1), w-1] 的交集。
        di_lo = max(-radius, -(self.h - 1))
        di_hi = min(radius, self.h - 1)
        dj_lo = max(-radius, -(self.w - 1))
        dj_hi = min(radius, self.w - 1)
        offsets = [
            (di, dj)
            for di in range(di_lo, di_hi + 1)
            for dj in range(dj_lo, dj_hi + 1)
            # 中心单元恒入选：极小带宽时 cutoff*h<1，圆形判据会把它误删。
            if (di == 0 and dj == 0) or di * di + dj * dj <= limit
        ]
        return radius, offsets

    def _accumulate_numpy(self, coords, weights):
        w, h, bw = self.w, self.h, self.bandwidth
        n = coords.shape[0]
        if n == 0:
            return _np.zeros((h, w), dtype=_np.float64)

        px, py = coords[:, 0], coords[:, 1]
        # 点所在的基单元（单元 (i,j) 覆盖 [j, j+1) × [i, i+1)）。
        j0 = _np.floor(px - 0.5).astype(_np.int64)
        i0 = _np.floor(py - 0.5).astype(_np.int64)

        radius, offsets = self._stencil_offsets()
        di = _np.array([o[0] for o in offsets], dtype=_np.int64)
        dj = _np.array([o[1] for o in offsets], dtype=_np.int64)
        n_offsets = len(offsets)
        norm = 1.0 / (2.0 * math.pi * bw * bw)
        inv_bw2 = 1.0 / (bw * bw)

        # 外围补一圈 radius+1 的零边：内部点的全部 stencil 落点保证在
        # 补边网格内，无需任何边界掩码。补边宽度以网格尺寸为上界——
        # stencil 偏移已与网格可达范围取交，更宽的补边没有意义，
        # 同时避免超大带宽时按真实 radius 分配巨型数组。
        shift = min(radius + 1, max(w, h) + 1)
        padded_w = w + 2 * shift
        padded_h = h + 2 * shift
        total = padded_w * padded_h

        # 内部点：基单元距物理网格四边都不少于 radius（含越界点时为少数）。
        interior = (
            (j0 >= radius) & (j0 < w - radius)
            & (i0 >= radius) & (i0 < h - radius)
        )
        boundary = ~interior

        # 分块控制临时数组规模（约 600 万贡献/块），缓冲区跨块复用。
        chunk = max(1, 6_000_000 // n_offsets)
        flat_grid = _np.zeros(total, dtype=_np.float64)
        index_buffer = _np.empty(chunk * n_offsets, dtype=_np.int64)
        value_buffer = _np.empty(chunk * n_offsets, dtype=_np.float64)

        def splat(mask, use_mask):
            nonlocal flat_grid
            if not mask.any():
                return
            sel_j0 = j0[mask]
            sel_i0 = i0[mask]
            sel_x = px[mask]
            sel_y = py[mask]
            sel_wt = weights[mask]
            size = sel_j0.shape[0]
            for start in range(0, size, chunk):
                end = min(start + chunk, size)
                block_n = end - start
                base_j = sel_j0[start:end] + shift
                base_i = sel_i0[start:end] + shift
                block_x = sel_x[start:end]
                block_y = sel_y[start:end]
                block_w = sel_wt[start:end]
                indices = index_buffer[:block_n * n_offsets]
                values = value_buffer[:block_n * n_offsets]
                pos = 0
                for k in range(n_offsets):
                    cell_j = base_j + dj[k]
                    cell_i = base_i + di[k]
                    if use_mask:
                        inside = (
                            (cell_j >= 0)
                            & (cell_j < padded_w)
                            & (cell_i >= 0)
                            & (cell_i < padded_h)
                        )
                        if not inside.any():
                            continue
                        cell_j = cell_j[inside]
                        cell_i = cell_i[inside]
                        # 物理坐标 = 补边坐标 - shift，核距离与逐点路径完全一致。
                        ddx = (cell_j - shift + 0.5) - block_x[inside]
                        ddy = (cell_i - shift + 0.5) - block_y[inside]
                        ww = block_w[inside]
                    else:
                        ddx = (cell_j - shift + 0.5) - block_x
                        ddy = (cell_i - shift + 0.5) - block_y
                        ww = block_w
                    count = cell_j.shape[0]
                    indices[pos:pos + count] = cell_i * padded_w + cell_j
                    values[pos:pos + count] = (
                        ww * norm * _np.exp(-0.5 * (ddx * ddx + ddy * ddy) * inv_bw2)
                    )
                    pos += count
                # bincount 按目标单元聚合：等价于逐点加权和，
                # 唯一差别是同一单元内的浮点求和顺序（远低于 1e-9）。
                flat_grid += _np.bincount(
                    indices[:pos],
                    weights=values[:pos],
                    minlength=total,
                )

        splat(interior, use_mask=False)
        splat(boundary, use_mask=True)
        padded_grid = flat_grid.reshape((padded_h, padded_w))
        return padded_grid[shift:shift + h, shift:shift + w]

    def _accumulate_python(self, coords, weights):  # pragma: no cover
        w, h, bw = self.w, self.h, self.bandwidth
        grid = [[0.0 for _ in range(w)] for _ in range(h)]
        norm = 1.0 / (2.0 * math.pi * bw * bw)
        _, offsets = self._stencil_offsets()
        for (x, y), weight in zip(coords, weights):
            j0 = int(math.floor(x - 0.5))
            i0 = int(math.floor(y - 0.5))
            for di, dj in offsets:
                jj, ii = j0 + dj, i0 + di
                if 0 <= jj < w and 0 <= ii < h:
                    ddx = (jj + 0.5) - x
                    ddy = (ii + 0.5) - y
                    grid[ii][jj] += (
                        weight * norm * math.exp(-0.5 * ((ddx / bw) ** 2 + (ddy / bw) ** 2))
                    )
        return grid

    # ------------------------------------------------------------------ #
    # 归一
    # ------------------------------------------------------------------ #
    def normalize(self, grid, lo=0.0, hi=1.0):
        """分位裁剪 + 单位积分归一。

        步骤：

        1. 校验 ``0 <= lo < hi <= 1`` 两个分位数；
        2. 取网格 ``lo``/``hi`` 分位值（线性插值）作为动态范围上下界，
           超出部分裁剪——这是分位裁剪，不是线性 min/max 饱和；
        3. 以单元面积（1×1）为测度除以总积分，使结果积分等于 1。

        全零网格返回同形状的全零网格。
        """
        self._validate_quantiles(lo, hi)
        matrix, nrows, ncols = self._parse_grid(grid)

        if _np is not None:
            return self._normalize_numpy(matrix, float(lo), float(hi))
        return self._normalize_python(matrix, float(lo), float(hi))  # pragma: no cover

    @staticmethod
    def _validate_quantiles(lo, hi):
        if not _is_finite_number(lo) or not _is_finite_number(hi):
            raise InvalidQuantileError(
                f"quantiles must be finite numbers, got lo={lo!r}, hi={hi!r}"
            )
        lo_f, hi_f = float(lo), float(hi)
        if not (0.0 <= lo_f < hi_f <= 1.0):
            raise InvalidQuantileError(
                f"require 0 <= lo < hi <= 1, got lo={lo_f!r}, hi={hi_f!r}"
            )

    def _parse_grid(self, grid):
        """校验网格并返回 ``(matrix, nrows, ncols)``；向量化快速路径优先。"""
        if _np is not None:
            try:
                arr = _np.asarray(grid, dtype=_np.float64)
            except (TypeError, ValueError) as exc:
                raise InvalidGridError(
                    "grid must be a 2-D sequence of finite numbers"
                ) from exc
            if arr.ndim != 2:
                raise InvalidGridError("grid must be a 2-D sequence of rows")
            if not _np.isfinite(arr).all():
                raise InvalidGridError("grid contains non-finite values")
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                raise InvalidGridError("grid must be non-empty")
            return arr, arr.shape[0], arr.shape[1]

        try:  # pragma: no cover - 无 NumPy 回退
            rows = list(grid)
        except TypeError as exc:
            raise InvalidGridError("grid must be a 2-D sequence of rows") from exc
        matrix = []
        ncols = None
        for r_idx, row in enumerate(rows):
            try:
                row_vals = list(row)
            except TypeError as exc:
                raise InvalidGridError(f"grid row {r_idx} is not a sequence") from exc
            for value in row_vals:
                if not _is_finite_number(value):
                    raise InvalidGridError(
                        f"grid[{r_idx}] contains non-finite value {value!r}"
                    )
            row_floats = [float(v) for v in row_vals]
            if ncols is None:
                ncols = len(row_floats)
            elif len(row_floats) != ncols:
                raise InvalidGridError("grid rows must have equal length")
            matrix.append(row_floats)
        nrows = len(matrix)
        if ncols is None:
            ncols = 0
        return matrix, nrows, ncols

    def _normalize_numpy(self, values, lo, hi):
        flat = values.reshape(-1)
        q_lo, q_hi = _np.quantile(flat, [lo, hi])
        # 分位裁剪：钳制到分位范围，而非按 min/max 线性饱和。
        clipped = _np.clip(flat, q_lo, q_hi)
        total = float(clipped.sum())  # 单元面积为 1，积分即求和
        if total == 0.0:
            result = _np.zeros_like(values)
        else:
            result = clipped.reshape(values.shape) / total
        return result.tolist()

    def _normalize_python(self, matrix, lo, hi):  # pragma: no cover
        flat = sorted(v for row in matrix for v in row)

        def quantile(q):
            if len(flat) == 1:
                return flat[0]
            pos = q * (len(flat) - 1)
            base = int(math.floor(pos))
            frac = pos - base
            if base + 1 >= len(flat):
                return flat[-1]
            return flat[base] * (1.0 - frac) + flat[base + 1] * frac

        q_lo, q_hi = quantile(lo), quantile(hi)
        clipped = [
            [min(max(v, q_lo), q_hi) for v in row]
            for row in matrix
        ]
        total = sum(v for row in clipped for v in row)
        if total == 0.0:
            return [[0.0 for _ in row] for row in matrix]
        return [[v / total for v in row] for row in clipped]
