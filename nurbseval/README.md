# nurbseval

## 题面（可直接粘贴）

子目录：nurbseval。NURBS 曲线求值。约束：节点向量必须非降且长度=控制点+次数+1（否则报错）；有理基函数之和为 1（≤1e-12）；端点处曲线经过首末控制点（夹持时）；一阶导数用差分对照（≤1e-6）；退化权重（全 0）报错；1 万次求值不超过 300 ms；并发安全。验收：cd nurbseval && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd nurbseval && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
