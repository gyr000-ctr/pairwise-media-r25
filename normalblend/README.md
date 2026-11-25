# normalblend

## 题面（可直接粘贴）

子目录：normalblend。法线贴图混合。约束：混合后必须重归一（≤1e-9）；叠加（whiteout/UDN）与参考式一致（≤1e-9）；法线需按切线基变换（TBN 正交性 ≤1e-9）；混合权重为 0/1 时退化为单输入；10 万像素不超过 400 ms；非法法线可区分报错。验收：cd normalblend && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd normalblend && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
