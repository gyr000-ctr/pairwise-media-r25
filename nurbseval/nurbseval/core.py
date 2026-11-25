"""nurbseval 核心 API（初始未实现）。"""


class Nurbs:
    def __init__(self, degree, ctrl, knots, weights):
        """构造"""
        raise NotImplementedError

    def eval(self, t):
        """求值"""
        raise NotImplementedError

    def derivative(self, t):
        """导数"""
        raise NotImplementedError
