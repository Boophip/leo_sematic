# Proxy-Guided Feasible Top-k PPO 主方法说明

报告日期：2026-06-24

## 1. 方法定位

从本阶段开始，`feasible-topk` 不再作为附属消融项处理，而是作为论文主方法中的候选动作生成模块。主方法建议命名为：

```text
Proxy-Guided Feasible Top-k PPO
```

核心叙事是：

```text
大规模物理合法动作空间
    -> 物理可行性过滤
    -> 质量代理与代价感知 top-k 候选生成
    -> PPO 在紧凑候选集合内学习长期调度
```

这样表述的原因是：星载在线调度的原始动作空间包含丢弃、本地多出口处理、以及多协作星、多出口、多压缩等级的跨星卸载组合。直接让 PPO 在完整动作空间中探索，学习难度高且样本效率差。`feasible-topk` 先用物理约束和轻量代理压缩候选集合，再让 PPO 学习队列、AoSI、时延和质量之间的长期取舍，更符合弱算力星载系统的工程约束。

## 2. 候选动作生成规则

每个 ROI 到来时，环境先枚举固定动作空间中的物理合法动作：

- `drop` 始终合法并保留。
- 本地动作仅允许 `compression_level=local`。
- 跨星卸载动作只允许选择当前可见且 SNR-AMC 支持的协作卫星。
- 不可见链路不进入候选集合，不能用大时延替代。

随后对合法的非丢弃动作打分并保留 top-k。当前实现使用：

```text
score = predicted_quality
        - delay_weight * normalized_delay
        - 0.05 * normalized_compressed_bytes
```

排序 tie-break 依次使用：

```text
predicted_quality 高优先
delay 低优先
compressed_bytes 低优先
固定动作索引稳定优先
```

最终 PPO 面对的动作集合为：

```text
{drop} union top-k legal non-drop actions
```

当前主配置为 `k=12`。

## 3. 与论文原文思路的关系

该改动不修改 `project.pdf` 中的研究公式和物理语义。它保留了原文中的关键设定：

- 源卫星侧先提取 ROI，而不是传输完整遥感大图。
- 在线调度使用离线 profiling 和 MLP 质量代理，不在 RL step 内运行视觉模型。
- 动作仍是本地早退或跨星完整卸载，不恢复模型中间切分。
- SNR 仍通过 AMC 查表，不做每步连续功率凸优化。
- AoSI、计算队列、质量/时延约束和 QoE 计算保持不变。

变化点只在于策略求解层：从“PPO 直接在完整合法动作集合中学习”调整为“代理引导的候选生成 + PPO 候选内选择”。这属于星载弱算力场景下的动作空间约简机制，而不是研究公式的静默修改。

## 4. 实验口径

主结果表应使用：

```text
strict proxy + feasible-topk(k=12) + PPO
```

当前最强配置是：

```text
strict proxy + feasible-topk(k=12) + shaping + 100k
```

对应产物：

```text
outputs/rl/ppo_topk_strict_proxy_shaping_100k/summary.json
outputs/rl/ppo_topk_strict_proxy_shaping_100k/comparison_report.md
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/
outputs/rl/ppo_topk_strict_proxy_no_shaping_100k/summary.json
```

`legal` 全动作空间 PPO 不再作为主方法，而作为消融/压力对照：

```text
Full legal action-space PPO
```

它的作用是说明：在不做候选生成时，PPO 虽然不会执行物理非法动作，但探索空间过宽，难以稳定超过强启发式基线。

## 5. 必须保留的风险说明

论文写作中需要明确以下边界：

- `feasible-topk` 是主方法的一部分，不能把结果表述为 full legal action-space PPO 的胜利。
- 候选排序只使用在线可得状态、物理链路状态、profiling 表和 strict proxy 预测，不使用 `task_quality`、`quality_label`、`predicted_class_id`、`exit_confidence` 等泄漏字段。
- 当前链路仍是可复现 synthetic visibility/SNR trace，不能写成 STK/TLE/ns-3 高保真在轨实验。
- 当前实现是单智能体 PPO，不应声称已经完成 MADRL。

## 6. 论文建议表述

英文方法句：

```text
We propose a proxy-guided feasible top-k candidate generation mechanism that
first filters physically invalid actions and then ranks feasible local/offloading
actions using proxy-estimated task quality, delay, and communication cost. The
PPO policy selects from this compact candidate set to optimize long-term queue,
delay, energy, and AoSI-aware objectives.
```

中文方法句：

```text
本文引入代理引导的可行 top-k 候选动作生成机制，先过滤不可见链路等物理非法动作，再基于质量代理预测、时延和通信数据量对合法动作排序，保留少量高潜力候选。PPO 策略仅在该紧凑候选集合中进行选择，从而提升星载弱算力场景下的学习效率和在线决策可行性。
```
