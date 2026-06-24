# Paper-facing 主方法结果报告

报告日期：2026-06-24

## 1. 报告目的

本报告把当前可用于论文实验章节的主方法结果统一整理出来。报告不修改研究公式、QoE 定义、AoSI 定义、动作语义或链路模型，只固定当前论文口径：

```text
Proxy-Guided Feasible Top-k PPO
```

主方法由两层组成：

1. 代理引导的可行 top-k 候选动作生成：过滤物理非法动作，并按质量代理、时延和通信字节数筛选候选。
2. PPO 调度策略：在紧凑候选集合内学习长期质量、时延、能耗、队列和 AoSI 权衡。

这意味着当前正向结果不能表述为 full legal action-space PPO 的胜利。full `legal` PPO 应作为动作空间压力对照，用来说明候选生成机制的必要性。

## 2. 实验边界

主结果使用：

```text
profile: data/profiling/dota_v1_lite_300_100_100/test_strict_image_features/roi_profile_smoke.csv
proxy: outputs/proxy/strict_image_features/quality_proxy.joblib
candidate mode: feasible-topk
candidate top-k: 12
seeds: 42, 43, 44
scenarios: default, low_snr, compute_congested, tight_deadline
limit_rois: 512
total_timesteps: 100000 per scenario
```

当前链路仍为可复现 synthetic visibility/SNR trace，不能写成 STK/TLE/ns-3 或真实在轨链路。当前实现仍是单智能体 PPO，不能写成已完成 MADRL。

## 3. Strict Proxy 结果

Strict profiling 按 DOTA 原始 image-level split 生成 train/val/test profiling 表，image_id 无重叠。当前 primary proxy 是轻量 MLP。

| Metric | Val | Test |
|---|---:|---:|
| MAE | 0.0905 | 0.0850 |
| MSE | 0.0253 | 0.0234 |
| R2 | 0.7822 | 0.8023 |
| High-value MAE | 0.0899 | 0.0846 |
| Threshold F1 | 0.9218 | 0.9374 |
| Top-1 Agreement | 0.2175 | 0.2467 |
| Mean Regret | 0.0431 | 0.0348 |

模型大小为 `529519` bytes。输入继续排除泄漏字段：

```text
quality_label
task_quality
predicted_class_id
exit_confidence
```

论文表述要点：proxy 的 R2 和 threshold F1 已足以支撑在线质量估计，但 top-1 agreement 不是 oracle 级别，因此主方法依赖候选约束和 PPO 学习共同完成调度，而不是只按 proxy 排序直接贪心执行。

## 4. 主结果总表

主结果配置：

```text
strict proxy + feasible-topk(k=12) + shaping + 100k
```

| Policy | Mean Rank | Best Count | Mean QoE | Mean Gap | Mean Success | Mean Delay ms | Mean Energy J | Mean AoSI Cost | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed-RL | 1.0000 | 12 / 12 | -2.0784 | 0.0000 | 0.9583 | 63.2113 | 2.1790 | 115.4084 | 0.0000 |
| No-AoSI | 2.7500 | 0 / 12 | -10.9356 | -8.8573 | 0.9092 | 108.9976 | 1.9747 | 122.1974 | 0.0000 |
| Semantic-Greedy | 4.1667 | 0 / 12 | -16.3247 | -14.2464 | 0.8608 | 113.4274 | 2.2117 | 137.3001 | 0.0000 |
| AoSI-Greedy | 4.0000 | 0 / 12 | -23.3523 | -21.2739 | 0.8408 | 178.5741 | 2.2736 | 127.0284 | 0.0000 |
| Best-SNR | 5.0000 | 0 / 12 | -27.9811 | -25.9027 | 0.8247 | 214.7504 | 2.4734 | 133.2012 | 0.0000 |
| Local-Adaptive | 5.2500 | 0 / 12 | -30.7827 | -28.7043 | 0.7822 | 165.0922 | 1.8416 | 157.0058 | 0.0000 |
| Local-Deep | 6.5000 | 0 / 12 | -40.4627 | -38.3844 | 0.7183 | 214.3233 | 2.4029 | 174.1017 | 0.0000 |
| Random | 7.3333 | 0 / 12 | -49.5800 | -47.5016 | 0.5509 | 157.7796 | 2.5002 | 200.4613 | 0.0000 |

主结论：Proposed-RL 在 3 seeds x 4 scenarios 的 12 个样本中全部排名第一，且无 executed illegal action。相比最强短视基线 No-AoSI，Mean QoE 提升 `+8.8573`，Mean Success 从 `0.9092` 提升到 `0.9583`，Mean Delay 从 `108.9976 ms` 降到 `63.2113 ms`。

## 5. 场景级结果

| Scenario | Best Policy | Best Mean QoE | Proposed-RL QoE | Proposed-RL Rank | Gap |
|---|---|---:|---:|---:|---:|
| compute_congested | Proposed-RL | -6.2276 | -6.2276 | 1.0000 | 0.0000 |
| default | Proposed-RL | -0.6667 | -0.6667 | 1.0000 | 0.0000 |
| low_snr | Proposed-RL | -1.0817 | -1.0817 | 1.0000 | 0.0000 |
| tight_deadline | Proposed-RL | -0.3374 | -0.3374 | 1.0000 | 0.0000 |

论文表述要点：主方法在默认、低信噪比、计算拥塞和严格 deadline 四类 deterministic stress scenarios 下均为最优策略。

## 6. Reward Shaping 消融

shaping 只影响 PPO 训练奖励，不改变最终报告的 canonical QoE。

| Run | Mean Rank | Best Count | Mean QoE | Mean Gap | Mean Success | Mean Delay ms | Mean AoSI Cost | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-k 100k no-shaping | 1.0833 | 11 / 12 | -2.1354 | -0.0260 | 0.9478 | 62.8920 | 116.8893 | 0.0000 |
| Top-k 100k shaping | 1.0000 | 12 / 12 | -2.0784 | 0.0000 | 0.9583 | 63.2113 | 115.4084 | 0.0000 |

结论：no-shaping 100k 已经明显优于强基线，但 shaping 在 Mean Rank、Best Count、Mean QoE、Mean Success 和 AoSI 上更稳。论文主结果保留 shaping，no-shaping 作为训练奖励消融。

## 7. 候选动作机制必要性

full `legal` action-space PPO 的已有对照不与 strict 100k 主表做严格同口径数值比较，因为它使用较早的 proxy/训练预算。但它可作为动作空间压力证据：不做 top-k 候选生成时，PPO 在较宽动作空间中难以稳定超过强启发式基线。

| Control Run | Candidate Mode | Timesteps | Proposed-RL Mean Rank | Proposed-RL Best Count | Proposed-RL Mean QoE | Proposed-RL Mean Success | Executed Illegal |
|---|---|---:|---:|---:|---:|---:|---:|
| lightweight proxy legal control | legal | 20000 | 4.3333 | 0 / 12 | -16.1886 | 0.8309 | 0.0000 |
| earlier legal formal | legal | 5000 | 4.3333 | 0 / 12 | -14.3282 | 0.8449 | 0.0000 |
| strict proxy main | feasible-topk | 100000 | 1.0000 | 12 / 12 | -2.0784 | 0.9583 | 0.0000 |

论文表述要点：`legal` 对照证明物理非法动作已被控制，但完整合法动作空间过宽，探索效率不足。`feasible-topk` 不是结果包装，而是星载在线调度所需的候选动作生成机制。

## 8. 图表清单

主结果图表已由 formal runner 生成：

```text
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/qoe_by_policy.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/qoe_by_scenario_policy.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/qoe_cdf_by_policy.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/training_curve_qoe.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/training_curve_actions.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/training_curve_violations.png
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/training_curve_queues.png
```

建议论文图表顺序：

1. QoE by Policy：主结果柱状图。
2. Scenario x Policy Mean QoE：四场景鲁棒性热力图。
3. QoE CDF by Policy：策略稳定性分布。
4. Training Evaluation Curve：PPO 收敛趋势。
5. Action Mix During Training：说明候选集内动作选择逐步稳定。
6. Violations / Virtual Queues：说明 constraint-aware shaping 的训练效果。

## 9. 可写入论文的 claims

可以写：

- 本文提出 Proxy-Guided Feasible Top-k PPO，将物理可行性过滤、质量代理估计和 PPO 长期调度结合起来。
- strict image-level split 下的轻量 MLP proxy 在 test split 上达到 `R2=0.8023`、`MAE=0.0850`。
- 在 synthetic deterministic stress scenarios 下，主方法在 12/12 seed-scenario 样本中取得最佳 QoE，且无 executed illegal action。
- reward shaping 提升了主方法的稳定性，100k shaping 优于 100k no-shaping。
- full legal action-space PPO 对照显示，候选动作生成对大动作空间下的 PPO 学习效率是关键。

不应写：

- 不应声称结果来自 STK/TLE/ns-3 或真实在轨链路。
- 不应声称已经完成 MADRL。
- 不应把 top-k 结果写成 full legal action-space PPO 结果。
- 不应把 deterministic sanity 的全量 ROI baseline 与 PPO formal 的 512 ROI 多 seed 结果混成同一张主表。

## 10. 下一步缺口

最值得补的是 k 消融：

```text
k = 4, 8, 12, 16
```

建议先用 20k 或 50k 做趋势验证，再决定是否需要 100k。训练前应按配置给出预估时间。该消融可以回答 “k=12 是否拍脑袋选择” 的审稿风险。
