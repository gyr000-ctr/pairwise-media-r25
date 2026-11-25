"""heatmap 核心 API（初始未实现）。"""


class Heatmap:
    def __init__(self, w, h, bandwidth):
        """构造"""
        raise NotImplementedError

    def accumulate(self, points):
        """累积"""
        raise NotImplementedError

    def normalize(self, grid, lo=0.0, hi=1.0):
        """归一"""
        raise NotImplementedError
