"""2D 场景图变换层级。

约定：
- 矩阵为 3x3 二维列表，元素按行存放；``compose(parent, child)`` 按列主序
  （列向量约定）合成，即 ``parent @ child``，与逐级 ``world = world @ local``
  的结果逐元素一致。
- 每个节点持有局部矩阵；世界矩阵沿 root -> node 路径懒计算并缓存，
  局部更新仅使本子树的缓存失效（写入为 O(1)，不遍历子树）。
- 所有公开方法在同一把可重入锁下执行，支持多线程并发读。
"""

import math
import threading

__all__ = [
    "Scene2D",
    "SceneGraphError",
    "DuplicateNodeError",
    "NodeNotFoundError",
    "CycleError",
    "DepthLimitError",
    "NonFiniteMatrixError",
    "SingularMatrixError",
]


class SceneGraphError(ValueError):
    """场景图相关错误的基类（均为 :class:`ValueError`）。"""


class DuplicateNodeError(SceneGraphError):
    """节点名重复。"""


class NodeNotFoundError(SceneGraphError):
    """引用了不存在的节点。"""


class CycleError(SceneGraphError):
    """重挂载会形成循环引用。"""


class DepthLimitError(SceneGraphError):
    """层级深度超过 ``max_depth`` 上限。"""


class NonFiniteMatrixError(SceneGraphError):
    """矩阵含 NaN/Inf 或形状不合法。"""


class SingularMatrixError(SceneGraphError):
    """矩阵不可逆（行列式为零或接近零）。"""


def _check_matrix(m):
    """校验并复制 3x3 矩阵。

    形状非法抛 :class:`SceneGraphError`；元素非数值或含 NaN/Inf 抛
    :class:`NonFiniteMatrixError`（两者均为 :class:`ValueError` 子类）。
    """
    if not isinstance(m, (list, tuple)) or len(m) != 3:
        raise SceneGraphError("矩阵必须是 3x3 的二维序列")
    rows = []
    for row in m:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise SceneGraphError("矩阵必须是 3x3 的二维序列")
        out_row = []
        for v in row:
            # bool 是 int 的子类，这里显式拒绝，避免 True/False 静默当 1/0
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise NonFiniteMatrixError("矩阵元素必须是数值类型")
            fv = float(v)
            if not math.isfinite(fv):
                raise NonFiniteMatrixError("矩阵含非有限元素 (NaN/Inf)")
            out_row.append(fv)
        rows.append(out_row)
    return rows


def _matmul(a, b):
    """3x3 矩阵乘法 ``a @ b``（列主序合成）。

    2D 齐次变换矩阵底行恒为 ``[0, 0, 1]``，对这类仿射矩阵走 2x3
    快路径；检测到底行非标准时回退到完整通用乘法，保证任意合法
    3x3 输入（含投影行）结果都与朴素定义一致。
    """
    if (
        a[2][0] == 0.0 and a[2][1] == 0.0 and a[2][2] == 1.0
        and b[2][0] == 0.0 and b[2][1] == 0.0 and b[2][2] == 1.0
    ):
        a00, a01, a02 = a[0]
        a10, a11, a12 = a[1]
        b00, b01, b02 = b[0]
        b10, b11, b12 = b[1]
        return [
            [
                a00 * b00 + a01 * b10,
                a00 * b01 + a01 * b11,
                a00 * b02 + a01 * b12 + a02,
            ],
            [
                a10 * b00 + a11 * b10,
                a10 * b01 + a11 * b11,
                a10 * b02 + a11 * b12 + a12,
            ],
            [0.0, 0.0, 1.0],
        ]
    return _matmul_generic(a, b)


def _matmul_generic(a, b):
    """完整 3x3 通用乘法。"""
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


_IDENTITY = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


class _Node:
    __slots__ = (
        "name",
        "parent",
        "children",
        "local",
        "depth",
        "rev",
        "_cache_rev",
        "_cache_world",
    )

    def __init__(self, name, parent, depth):
        self.name = name
        self.parent = parent
        self.children = []
        self.local = [row[:] for row in _IDENTITY]
        self.depth = depth
        # 局部修订号：每次 set_local / 重挂载自增
        self.rev = 0
        # 世界矩阵缓存及其对应的本节点修订号；祖先失效沿链逐级传播
        self._cache_rev = None
        self._cache_world = None


class Scene2D:
    """2D 场景图。

    构造时自动创建根节点 ``'root'``（深度 0，局部为单位矩阵）。
    ``max_depth`` 为允许的最大层数（根占第 1 层），例如 ``max_depth=1``
    时只能存在根节点，向根下挂子节点即报 :class:`DepthLimitError`；
    默认 64 即节点最大 0-based 深度为 63。
    """

    def __init__(self, max_depth=64):
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise ValueError("max_depth 必须是正整数")
        if max_depth < 1:
            raise ValueError("max_depth 必须是正整数")
        self._max_depth = max_depth
        self._lock = threading.RLock()
        self._nodes = {}
        root = _Node("root", None, 0)
        self._nodes["root"] = root

    # ------------------------------------------------------------------
    # 结构管理
    # ------------------------------------------------------------------
    def add(self, name, parent=None):
        """添加节点。

        - ``name`` 已存在：抛 :class:`DuplicateNodeError`（无论是否给
          ``parent``，重挂载请用 :meth:`reparent`）。
        - 新节点未给 ``parent`` 时默认挂到 ``'root'`` 下。
        """
        with self._lock:
            if not isinstance(name, str) or not name:
                raise NodeNotFoundError("节点名必须是非空字符串")
            if name in self._nodes:
                raise DuplicateNodeError("节点已存在: %r" % (name,))
            parent_name = "root" if parent is None else parent
            parent_node = self._nodes.get(parent_name)
            if parent_node is None:
                raise NodeNotFoundError("父节点不存在: %r" % (parent_name,))
            if parent_node.depth + 1 >= self._max_depth:
                raise DepthLimitError(
                    "节点 %r 的深度 %d 超过上限 %d"
                    % (name, parent_node.depth + 1, self._max_depth - 1)
                )
            node = _Node(name, parent_node, parent_node.depth + 1)
            parent_node.children.append(node)
            self._nodes[name] = node

    def reparent(self, name, parent):
        """把已存在节点 ``name`` 重新挂载到 ``parent`` 下。"""
        with self._lock:
            node = self._nodes.get(name)
            if node is None:
                raise NodeNotFoundError("节点不存在: %r" % (name,))
            if name == "root":
                raise CycleError("根节点不能被重新挂载")
            self._reparent(node, parent)

    def _reparent(self, node, parent_name):
        if not isinstance(parent_name, str):
            raise NodeNotFoundError("父节点名必须是字符串")
        new_parent = self._nodes.get(parent_name)
        if new_parent is None:
            raise NodeNotFoundError("父节点不存在: %r" % (parent_name,))
        if new_parent is node:
            raise CycleError("不能把节点挂载到自身: %r" % (node.name,))
        if node.parent is new_parent:
            return

        # 循环检测：新父节点不得位于 node 的子树中
        ancestor = new_parent
        while ancestor is not None:
            if ancestor is node:
                raise CycleError(
                    "重挂载 %r -> %r 会形成循环引用"
                    % (node.name, parent_name)
                )
            ancestor = ancestor.parent

        # 深度校验：node 子树的最深深度不得超过上限（一次遍历求最深值）
        max_sub_depth = node.depth
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.depth > max_sub_depth:
                max_sub_depth = cur.depth
            stack.extend(cur.children)
        shift = new_parent.depth + 1 - node.depth
        if max_sub_depth + shift >= self._max_depth:
            raise DepthLimitError(
                "重挂载后子树深度将超过上限 %d" % (self._max_depth,)
            )

        # 执行迁移：摘子节点、更新深度、自增修订号使子树缓存全部失效
        old_parent = node.parent
        if old_parent is not None:
            old_parent.children.remove(node)
        node.parent = new_parent
        new_parent.children.append(node)
        node.depth += shift
        node.rev += 1
        stack = list(node.children)
        while stack:
            cur = stack.pop()
            cur.depth = cur.parent.depth + 1
            cur.rev += 1
            stack.extend(cur.children)

    # ------------------------------------------------------------------
    # 局部 / 世界变换
    # ------------------------------------------------------------------
    def set_local(self, name, m):
        """设置节点的局部矩阵（仅影响该节点子树的世界矩阵）。"""
        local = _check_matrix(m)
        with self._lock:
            node = self._nodes.get(name)
            if node is None:
                raise NodeNotFoundError("节点不存在: %r" % (name,))
            node.local = local
            node.rev += 1
            # 本节点缓存立即失效；子孙缓存通过 rev 不匹配在下次 world()
            # 读取时沿链惰性检出，写入本身 O(1)、不遍历子树
            node._cache_rev = None
            node._cache_world = None

    def world(self, name):
        """返回节点的世界矩阵（新副本，调用方可自由修改）。

        沿 root -> node 路径逐级 ``world = world @ local`` 合成。
        缓存按各节点的局部修订号逐级校验：路径上任何节点的局部矩阵
        变化都会自动使其下游缓存失效。缓存命中的热路径只沿
        ``parent`` 指针做标量比较、不分配临时容器；确认失效后才
        收集路径并重算受影响的下游段。
        """
        with self._lock:
            node = self._nodes.get(name)
            if node is None:
                raise NodeNotFoundError("节点不存在: %r" % (name,))

            # 热路径：沿 parent 链向上校验，全部有效则直接复制缓存
            cur = node
            while cur is not None:
                if cur._cache_rev != cur.rev or cur._cache_world is None:
                    break
                cur = cur.parent
            else:
                return [row[:] for row in node._cache_world]

            # 存在失效节点：收集 root -> node 路径
            chain = []
            cur = node
            while cur is not None:
                chain.append(cur)
                cur = cur.parent
            chain.reverse()

            # 从 root 向下定位第一个失效点
            start = len(chain)
            for idx, n in enumerate(chain):
                if n._cache_rev != n.rev or n._cache_world is None:
                    start = idx
                    break

            if start == 0:
                # 链头失效：其世界矩阵就是自身局部矩阵
                world = [row[:] for row in chain[0].local]
                head = chain[0]
                head._cache_world = world
                head._cache_rev = head.rev
                tail = chain[1:]
            else:
                # 从失效点的前一个有效缓存继续向下合成
                world = chain[start - 1]._cache_world
                tail = chain[start:]
            for n in tail:
                world = _matmul(world, n.local)
                n._cache_world = world
                n._cache_rev = n.rev
            return [row[:] for row in world]

    # ------------------------------------------------------------------
    # 矩阵工具
    # ------------------------------------------------------------------
    def inverse(self, m):
        """求 3x3 齐次矩阵的逆（伴随矩阵法），奇异时抛 :class:`SingularMatrixError`。"""
        mat = _check_matrix(m)
        a, b, c = mat[0]
        d, e, f = mat[1]
        g, h, i = mat[2]

        det = (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )
        scale = max(
            1.0,
            abs(a), abs(b), abs(c),
            abs(d), abs(e), abs(f),
            abs(g), abs(h), abs(i),
        ) ** 3
        if not math.isfinite(det) or abs(det) <= 1e-12 * scale:
            raise SingularMatrixError("矩阵奇异，无法求逆")

        inv_det = 1.0 / det
        return [
            [
                (e * i - f * h) * inv_det,
                (c * h - b * i) * inv_det,
                (b * f - c * e) * inv_det,
            ],
            [
                (f * g - d * i) * inv_det,
                (a * i - c * g) * inv_det,
                (c * d - a * f) * inv_det,
            ],
            [
                (d * h - e * g) * inv_det,
                (b * g - a * h) * inv_det,
                (a * e - b * d) * inv_det,
            ],
        ]

    def compose(self, parent, child):
        """列主序合成两个局部矩阵：``parent @ child``。"""
        p = _check_matrix(parent)
        c = _check_matrix(child)
        return _matmul(p, c)
