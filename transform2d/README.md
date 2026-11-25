# transform2d

## 题面（可直接粘贴）

子目录：transform2d。2D 场景图变换层级。约束：父子矩阵按列主序合成（与逐级相乘一致 ≤1e-12）；世界矩阵求逆往返 ≤1e-9；循环引用检测并报错；深度超过上限报错；局部变换更新只影响子树；10 万次更新不超过 400 ms；并发读安全。验收：cd transform2d && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd transform2d && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
