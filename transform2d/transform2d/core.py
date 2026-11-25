"""transform2d 核心 API（初始未实现）。"""


class Scene2D:
    def __init__(self, max_depth=64):
        """构造"""
        raise NotImplementedError

    def add(self, name, parent):
        """加节点"""
        raise NotImplementedError

    def set_local(self, name, m):
        """设局部"""
        raise NotImplementedError

    def world(self, name):
        """世界矩阵"""
        raise NotImplementedError
