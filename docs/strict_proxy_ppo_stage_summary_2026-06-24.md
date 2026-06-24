# Strict Proxy PPO 阶段性整理报告

报告日期：2026-06-24

## 1. 阶段结论

本阶段完成了从严格 train/val/test 隔离质量代理，到 strict proxy 下 feasible-topk PPO 20k/50k/100k formal run 的闭环验证。当前最强结果是：

```text
strict proxy + feasible-topk(k=12) + shaping + 100k
```

该设置在 3 个 seed、4 个 deterministic stress scenarios、512 ROI 的聚合评估中达到：

| Metric | Proposed-RL |
|---|---:|
| Mean Rank | 1.0000 |
| Best Count | 12 / 12 |
| Mean QoE | -2.0784 |
| Mean QoE Gap to Best | 0.0000 |
| Mean Success Rate | 0.9583 |
| Mean Semantic Success Rate | 0.9630 |
| Mean Delay ms | 63.2113 |
| Mean Energy J | 2.1790 |
| Mean AoSI Cost | 115.4084 |
| Executed Illegal Count | 0.0000 |

结论：100k shaping 通过当前 gate，可以进入论文级结果整理和图表生成阶段。

方法边界已经明确：`feasible-topk` 纳入论文主方法，作为 `Proxy-Guided Feasible Top-k PPO` 的候选动作生成模块。它不是 full legal action-space PPO 的静默替代，而是显式的物理可行性过滤与代理引导 top-k 候选约简。`legal` 全动作空间 PPO 保留为消融/压力对照，用于说明没有候选生成时 PPO 学习难度显著增大。

## 2. Strict Proxy 与数据边界

Strict profiling 使用原始 DOTA image-level split，对 train/val/test 分别生成 profiling 表：

| Split | Rows | Image Count |
|---|---:|---:|
| train | 386720 | 299 |
| val | 133120 | 100 |
| test | 136840 | 100 |

image_id overlap 检查结果：

| Overlap | Result |
|---|---|
| train-val | false |
| train-test | false |
| val-test | false |

Primary proxy:

| Item | Value |
|---|---:|
| Model | mlp |
| Model Size | 529519 bytes |
| Test MAE | 0.084992 |
| Test MSE | 0.023398 |
| Test R2 | 0.802319 |
| Test High-value MAE | 0.084650 |
| Test Threshold F1 | 0.937434 |
| Test Top-1 Agreement | 0.246711 |
| Test Mean Regret | 0.034807 |

泄漏字段继续禁止作为输入：

```text
quality_label
task_quality
predicted_class_id
exit_confidence
```

主要产物：

```text
outputs/proxy/strict_image_features/quality_proxy.joblib
outputs/proxy/strict_image_features/summary.json
```

## 3. PPO Formal 结果

所有 run 均使用：

```text
profile: data/profiling/dota_v1_lite_300_100_100/test_strict_image_features/roi_profile_smoke.csv
proxy: outputs/proxy/strict_image_features/quality_proxy.joblib
candidate mode: feasible-topk
candidate top-k: 12
seeds: 42, 43, 44
scenarios: default, low_snr, compute_congested, tight_deadline
limit_rois: 512
```

聚合结果：

| Run | Mean Rank | Best Count | Mean QoE | Mean Gap | Mean Success | Mean Delay ms | Mean AoSI Cost | Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k no-shaping | 1.5833 | 8 / 12 | -4.5904 | -1.3072 | 0.9102 | 70.3923 | 122.0366 | 0.0000 |
| 20k shaping | 1.5833 | 8 / 12 | -2.9935 | -0.3312 | 0.9520 | 64.9870 | 117.6277 | 0.0000 |
| 50k shaping | 1.1667 | 11 / 12 | -2.2025 | -0.0812 | 0.9574 | 63.3821 | 115.9574 | 0.0000 |
| 100k no-shaping | 1.0833 | 11 / 12 | -2.1354 | -0.0260 | 0.9478 | 62.8920 | 116.8893 | 0.0000 |
| 100k shaping | 1.0000 | 12 / 12 | -2.0784 | 0.0000 | 0.9583 | 63.2113 | 115.4084 | 0.0000 |

100k 场景级结果：

| Scenario | Best Policy | Proposed-RL QoE | Proposed-RL Rank | Gap |
|---|---|---:|---:|---:|
| compute_congested | Proposed-RL | -6.2276 | 1.0000 | 0.0000 |
| default | Proposed-RL | -0.6667 | 1.0000 | 0.0000 |
| low_snr | Proposed-RL | -1.0817 | 1.0000 | 0.0000 |
| tight_deadline | Proposed-RL | -0.3374 | 1.0000 | 0.0000 |

50k 到 100k 的变化：

| Metric | 50k | 100k | Direction |
|---|---:|---:|---|
| Mean Rank | 1.1667 | 1.0000 | better |
| Best Count | 11 / 12 | 12 / 12 | better |
| Mean QoE | -2.2025 | -2.0784 | better |
| Mean Gap | -0.0812 | 0.0000 | better |
| Mean Success | 0.9574 | 0.9583 | better |
| Low-SNR Gap | -0.2569 | 0.0000 | better |

100k shaping 与 100k no-shaping 对照：

| Metric | 100k no-shaping | 100k shaping | Direction |
|---|---:|---:|---|
| Mean Rank | 1.0833 | 1.0000 | shaping better |
| Best Count | 11 / 12 | 12 / 12 | shaping better |
| Mean QoE | -2.1354 | -2.0784 | shaping better |
| Mean Gap | -0.0260 | 0.0000 | shaping better |
| Mean Success | 0.9478 | 0.9583 | shaping better |
| Mean Delay ms | 62.8920 | 63.2113 | no-shaping slightly lower |
| Mean AoSI Cost | 116.8893 | 115.4084 | shaping better |

结论：no-shaping 100k 仍显著优于强基线，但略弱于 100k shaping。当前论文主结果建议保留 `strict proxy + feasible-topk(k=12) + shaping + 100k`，并将 no-shaping 100k 作为训练奖励消融。

主要产物：

```text
outputs/rl/ppo_topk_strict_proxy_no_shaping_20k/summary.json
outputs/rl/ppo_topk_strict_proxy_no_shaping_100k/summary.json
outputs/rl/ppo_topk_strict_proxy_shaping_20k/summary.json
outputs/rl/ppo_topk_strict_proxy_shaping_50k/summary.json
outputs/rl/ppo_topk_strict_proxy_shaping_100k/summary.json
outputs/rl/ppo_topk_strict_proxy_shaping_100k/comparison_report.md
outputs/rl/ppo_topk_strict_proxy_shaping_100k/figures/
```

## 4. Deterministic Sanity

使用 strict test profile 与 strict proxy 的 deterministic all-scenarios sanity 已通过非法动作检查：

```text
executed_illegal_total: 0
```

强基线排序中，deterministic policy 仍以 `No-AoSI` 为最优短视策略：

| Scenario | Best Deterministic Policy | QoE | Success |
|---|---|---:|---:|
| compute_congested | No-AoSI | -82.0925 | 0.9458 |
| default | No-AoSI | -55.1372 | 0.9550 |
| low_snr | No-AoSI | -54.1235 | 0.9535 |
| tight_deadline | No-AoSI | -276.2024 | 0.7910 |

该结果用于 sanity 与物理直觉核对，不与 PPO formal aggregate 混写为同一类结论。

主要产物：

```text
outputs/simulation/deterministic_strict_proxy_all/summary_all.json
```

## 5. 当前论文表述边界

可以写入阶段结论的内容：

- strict proxy 的训练、验证、测试边界已收紧到 image-level train/val/test split。
- primary proxy 是轻量 MLP，体积远低于 50 MB，test R2 约 0.8023。
- Proxy-Guided Feasible Top-k PPO 是当前主方法：先过滤物理非法动作，再按质量代理、时延和通信字节数保留 top-k 候选，最后由 PPO 在候选集合内学习长期调度。
- feasible-topk + shaping 的 PPO 学习曲线和多 seed formal 结果随 20k -> 50k -> 100k 稳定增强。
- 100k shaping 在当前 synthetic deterministic stress scenarios 下实现 12/12 best，并且无 executed illegal action。

暂时不应过度表述的内容：

- 不应把 `feasible-topk` 结果写成原始 full `legal` 全动作空间 PPO 的胜利。
- 不应把 synthetic visibility/SNR trace 表述为 STK/TLE/ns-3 高保真链路结果。
- 不应把 deterministic No-AoSI sanity 与 PPO formal result 混成同一评估口径。
- 不应声称已经完成 MADRL 或多智能体扩展。

建议论文方法章节表述：

```text
We propose a proxy-guided feasible top-k candidate generation mechanism that
first filters physically invalid actions and then ranks feasible local/offloading
actions using proxy-estimated task quality, delay, and communication cost. The
PPO policy selects from this compact candidate set to optimize long-term queue,
delay, energy, and AoSI-aware objectives.
```

## 6. 下一阶段建议

P0：论文级结果固化

- 汇总 strict proxy、deterministic sanity、20k/50k/100k PPO 和 no-shaping ablation。
- 生成 QoE、success、delay、energy、AoSI、CDF、training curve 的最终图表。
- 将 100k shaping 作为当前 top-k candidate-generation 路线的候选最终规模。

P1：方法边界落地

- 将 `feasible-topk` 写入方法章节，命名为 Proxy-Guided Feasible Top-k PPO。
- 明确候选集生成与 top-k 选择规则。
- 保留 `legal` 结果作为全动作空间学习困难的对照，不包装成主结果。
- 补跑 strict proxy + feasible-topk + no-shaping 的更大规模对照，确认 shaping/no-shaping 取舍。

P2：论文风险补强

- 补一段 strict split 防泄漏说明，明确 proxy selection 只看 val，test 只做 final report。
- 补一段 synthetic trace 限制说明，后续如有时间再接 STK/TLE/ns-3 链路。
- 检查 `low_snr` 从 20k 到 100k 的改善原因，避免只用单一 aggregate 讲故事。

## 7. 验证状态

最新全量单元测试：

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

结果：

```text
Ran 97 tests in 9.575s
OK
```

本阶段没有修改研究公式、QoE 定义、AoSI 定义或动作语义。
