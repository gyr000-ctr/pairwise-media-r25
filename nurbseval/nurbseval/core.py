"""NURBS 曲线求值（纯 Python 标准库实现，线程安全）。

公开 API：

* ``Nurbs(degree, ctrl, knots, weights)`` 构造一条 NURBS 曲线；
* :meth:`Nurbs.eval` 曲线求值，返回 tuple；
* :meth:`Nurbs.derivative` 一阶导数，返回 tuple；
* :meth:`Nurbs.basis` 全部控制点的有理基函数，返回 list（和为 1）。

实例构造后内部状态全部为不可变对象、方法无共享可变缓存，
因此多线程并发调用是安全的。
"""

import math

__all__ = ["Nurbs"]


class Nurbs:
    """p 次 NURBS 曲线。

    参数
    ----
    degree:
        曲线次数 p，正整数（p >= 1）。
    ctrl:
        控制点序列，长度 n >= p + 1，每个点维度相同且分量有限。
    knots:
        节点向量，长度必须为 n + p + 1，非降序且全部有限。
    weights:
        权重序列，长度 n，全部有限；全 0 视为退化，抛 ``ValueError``。
    """

    def __init__(self, degree, ctrl, knots, weights):
        # ---- 次数校验 ----
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise ValueError("degree 必须是整数")
        if degree <= 0:
            raise ValueError("degree 必须为正整数")
        p = degree

        # ---- 控制点校验 ----
        ctrl = list(ctrl)
        n = len(ctrl)
        if n < p + 1:
            raise ValueError("控制点数量必须满足 n >= degree + 1")
        dim = len(ctrl[0])
        if dim == 0:
            raise ValueError("控制点维度必须 >= 1")
        points = []
        for point in ctrl:
            if len(point) != dim:
                raise ValueError("所有控制点维度必须一致")
            row = []
            for x in point:
                fx = float(x)
                if not math.isfinite(fx):
                    raise ValueError("控制点分量必须有限")
                row.append(fx)
            points.append(tuple(row))

        # ---- 节点向量校验 ----
        knots = list(knots)
        if len(knots) != n + p + 1:
            raise ValueError(
                "节点向量长度必须等于 控制点数 + 次数 + 1 = %d，实际 %d"
                % (n + p + 1, len(knots))
            )
        ks = []
        prev = -math.inf
        for k in knots:
            fk = float(k)
            if not math.isfinite(fk):
                raise ValueError("节点必须有限")
            if fk < prev:
                raise ValueError("节点向量必须非降序")
            prev = fk
            ks.append(fk)

        # ---- 权重校验 ----
        weights = list(weights)
        if len(weights) != n:
            raise ValueError("权重数量必须与控制点数量一致")
        ws = []
        any_nonzero = False
        for w in weights:
            fw = float(w)
            if not math.isfinite(fw):
                raise ValueError("权重必须有限")
            if fw != 0.0:
                any_nonzero = True
            ws.append(fw)
        if not any_nonzero:
            raise ValueError("权重全为 0：退化的 NURBS 曲线")

        if ks[-1] <= ks[0]:
            raise ValueError("参数域长度为 0（首尾节点相同）")

        # 全部存为不可变对象；构造后实例不再有任何可变状态
        self._degree = p
        self._n = n
        self._dim = dim
        self._ctrl = tuple(points)
        self._knots = tuple(ks)
        self._weights = tuple(ws)
        self._u_min = ks[0]
        self._u_max = ks[-1]

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _check_t(self, t):
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise ValueError("参数 t 必须是实数")
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("参数 t 必须有限")
        if t < self._u_min or t > self._u_max:
            raise ValueError(
                "参数 t=%r 超出参数域 [%r, %r]" % (t, self._u_min, self._u_max)
            )
        return t

    def _find_span(self, t):
        """返回节点区间编号 k，使 knots[k] <= t < knots[k+1]（端点夹持）。"""
        knots = self._knots
        n = self._n
        p = self._degree
        if t <= knots[p]:
            return p
        if t >= knots[n]:
            return n - 1
        lo, hi = p, n
        while lo < hi:
            mid = (lo + hi) // 2
            if t < knots[mid]:
                hi = mid
            elif t >= knots[mid + 1]:
                lo = mid + 1
            else:
                return mid
        return lo

    def _basis_table(self, t, span, q):
        """Cox-de Boor：返回 span 处 q 次非零基函数值，长度 q + 1。

        返回值下标 r 对应全局基函数编号 ``span - q + r``。
        """
        knots = self._knots
        vals = [0.0] * (q + 1)
        vals[0] = 1.0
        left = [0.0] * (q + 1)
        right = [0.0] * (q + 1)
        for j in range(1, q + 1):
            left[j] = t - knots[span + 1 - j]
            right[j] = knots[span + j] - t
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = vals[r] / denom
                vals[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            vals[j] = saved
        return vals

    def _weighted_data(self, t, need_derivative):
        """计算 (有理基 R, 点 C, 导数 C' 或 None)。"""
        p = self._degree
        n = self._n
        span = self._find_span(t)
        nvals = self._basis_table(t, span, p)

        weights = self._weights
        ctrl = self._ctrl
        first = span - p  # 全局起始控制点编号

        # 权函数 w(t) = Σ N_i w_i
        wsum = 0.0
        for r in range(p + 1):
            wsum += nvals[r] * weights[first + r]
        if wsum == 0.0:
            raise ValueError("当前参数处有效权重之和为 0，曲线未定义")

        # 有理基与曲线点
        rational = [0.0] * n
        point = [0.0] * self._dim
        for r in range(p + 1):
            ri = nvals[r] * weights[first + r] / wsum
            rational[first + r] = ri
            cp = ctrl[first + r]
            for d in range(self._dim):
                point[d] += ri * cp[d]

        deriv = None
        if need_derivative:
            # 次数 p-1 的基函数，全局编号 span-p+1 .. span
            if p == 1:
                mvals = [1.0]
                m_first = span
            else:
                mvals = self._basis_table(t, span, p - 1)
                m_first = span - p + 1
            knots = self._knots

            w_prime = 0.0
            a_prime = [0.0] * self._dim
            for i in range(first, span + 1):
                dn = 0.0
                # +p * N_{i,p-1} / (U[i+p] - U[i])
                if i >= m_first:
                    mr = i - m_first
                    denom = knots[i + p] - knots[i]
                    if denom != 0.0:
                        dn += p * mvals[mr] / denom
                # -p * N_{i+1,p-1} / (U[i+p+1] - U[i+1])
                if i + 1 <= span:
                    mr = i + 1 - m_first
                    denom = knots[i + p + 1] - knots[i + 1]
                    if denom != 0.0:
                        dn -= p * mvals[mr] / denom
                wi = weights[i]
                w_prime += dn * wi
                cp = ctrl[i]
                for d in range(self._dim):
                    a_prime[d] += dn * wi * cp[d]

            # C' = (A' - C w') / w
            deriv = tuple(
                (a_prime[d] - point[d] * w_prime) / wsum
                for d in range(self._dim)
            )

        return rational, tuple(point), deriv

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def eval(self, t):
        """计算曲线点 C(t)，返回与控制点同维度的 tuple。"""
        t = self._check_t(t)
        _, point, _ = self._weighted_data(t, False)
        return point

    def derivative(self, t):
        """计算一阶导数 C'(t)，返回与控制点同维度的 tuple。"""
        t = self._check_t(t)
        _, _, deriv = self._weighted_data(t, True)
        return deriv

    def basis(self, t):
        """返回全部 n 个控制点的有理基函数（list，分量和为 1）。"""
        t = self._check_t(t)
        rational, _, _ = self._weighted_data(t, False)
        return rational
