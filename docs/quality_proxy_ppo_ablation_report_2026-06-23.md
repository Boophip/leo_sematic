# 轻量质量代理与 PPO 调度 Ablation 实验报告

报告日期：2026-06-23

## 1. 结论摘要

本轮实验在不修改研究公式、QoE、AoSI、PPO 环境接口的前提下，完成了新轻量质量代理接入后的 deterministic sanity check、3-seed PPO formal run，以及三组关键 ablation。当前结果支持进入下一阶段更大规模 formal，但建议把 `feasible-topk + 20k timesteps` 作为 PPO 的最低配置；`legal` 全动作空间和 5k timesteps 都不足以支撑稳定结论。

最强当前配置是：

```text
profile: data/profiling/dota_v1_lite_300_100_100/test_full_image_features/roi_profile_smoke.csv
proxy: outputs/proxy/lightweight_image_features/quality_proxy.joblib
candidate_mode: feasible-topk
candidate_top_k: 12
total_timesteps: 20000
reward_shaping: 0.0 / 0.0 / 0.0
seeds: 42, 43, 44
```

该配置下 `Proposed-RL` 在 12 个 seed-scenario 样本中均为第一，mean rank 为 `1.00`，mean QoE gap to best 为 `0.000`，mean success rate 为 `0.9196`，executed illegal count 为 `0`。不过 no-shaping 版本有少量 seed-scenario 的 success rate 低于 0.90，说明如果后续论文要求更强的逐 seed 鲁棒性，还需要把 checkpoint selection 或 success gate 做得更严格。

## 2. 实验边界

本报告只固化质量代理和调度实验结果，不修改以下内容：

- 质量代理研究公式：`q_k = Proxy(ROI metadata, exit_level, compression_level)`。
- ROI 检测器、多出口 CNN、QoE 公式、AoSI 公式。
- PPO 环境接口和动作含义。
- deterministic baseline 的策略定义。

新 proxy 使用的图像质量特征来自 ROI crop 或压缩解码后的 ROI 图像，属于在线可得或离线 profiling 可查信息；未把 `task_quality`、`quality_label`、`predicted_class_id`、`exit_confidence` 等泄漏字段作为输入。

## 3. 数据与输出清单

| 类型 | 路径 |
|---|---|
| 新 profiling | `data/profiling/dota_v1_lite_300_100_100/test_full_image_features/roi_profile_smoke.csv` |
| 轻量 proxy | `outputs/proxy/lightweight_image_features/quality_proxy.joblib` |
| proxy 对比 summary | `outputs/proxy/lightweight_image_features/summary.json` |
| deterministic sanity | `outputs/simulation/deterministic_lightweight_proxy_all/summary_all.json` |
| PPO 20k top-k shaping | `outputs/rl/ppo_topk_lightweight_proxy_formal_20k/summary.json` |
| PPO 5k top-k shaping | `outputs/rl/ppo_topk_lightweight_proxy_formal_5k/summary.json` |
| PPO 20k legal shaping | `outputs/rl/ppo_legal_lightweight_proxy_formal_20k/summary.json` |
| PPO 20k top-k no-shaping | `outputs/rl/ppo_topk_lightweight_proxy_no_shaping_20k/summary.json` |

训练前已给出保守估计：单次 20k formal ablation 约 24-96 分钟。实际本轮补跑中，`legal` 20k 用时约 452 秒，`top-k no-shaping` 20k 用时约 518 秒。

## 4. 质量代理结果

| Proxy 版本 | 输入特征 | 模型大小 | Test MAE | Test R2 | High-value MAE | Threshold F1 | Mean regret | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Metadata-only MLP | 表格元信息 | 0.06 MB | 0.1893 | 0.4607 | - | - | - | 旧 baseline，泛化偏弱 |
| Image-feature MLP 128/64/32 | 元信息 + ROI 图像质量特征 | 0.53 MB | 0.1344 | 0.6357 | 0.1333 | 0.8599 | 0.0691 | 当前轻量默认 |
| HistGBDT | 元信息 + ROI 图像质量特征 | 1.87 MB | 0.1497 | 0.6157 | 0.1526 | 0.8450 | 0.0741 | 轻量但弱于 MLP |
| ExtraTrees | 元信息 + ROI 图像质量特征 | 857.56 MB | 0.1339 | 0.6600 | 0.1355 | 0.8630 | 0.0670 | 指标略高，但不轻量 |

结论：图像质量特征带来主要收益，轻量 MLP 已经把 R2 从 `0.4607` 提升到 `0.6357`。ExtraTrees 指标略高，但体积接近 858 MB，不符合轻量代理目标，因此不作为当前默认。

## 5. Deterministic Sanity

使用新轻量 proxy 后，四场景 deterministic baseline 的聚合表现如下：

| Policy | Mean QoE | Success rate | Quality violation | Timeout | Executed illegal |
|---|---:|---:|---:|---:|---:|
| Semantic-Greedy | -6.150 | 0.9282 | 29.0 | 8.0 | 0 |
| No-AoSI | -8.654 | 0.9155 | 28.0 | 16.0 | 0 |
| AoSI-Greedy | -17.414 | 0.8804 | 28.0 | 35.3 | 0 |
| Local-Adaptive | -17.605 | 0.8613 | 41.0 | 32.0 | 0 |
| Best-SNR | -26.549 | 0.8291 | 52.0 | 38.5 | 0 |
| Local-Deep | -27.680 | 0.7886 | 52.0 | 62.5 | 0 |
| Random | -46.533 | 0.5708 | 169.0 | 34.3 | 0 |

这说明新 proxy 下 deterministic sanity check 正常：语义贪心和 No-AoSI 仍然是强基线，后续 PPO 必须接近或超过这两者才有意义。

## 6. PPO Formal 与 Ablation

`Proposed-RL` 聚合指标：

| 实验 | Timesteps | Candidate | Shaping 权重 | Mean rank | Best count | Mean QoE | Gap to best | Success | Quality violation | Timeout | Illegal |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-k 20k no-shaping | 20000 | feasible-topk, k=12 | 0 / 0 / 0 | 1.00 | 12/12 | -2.845 | 0.000 | 0.9196 | 27.5 | 2.0 | 0 |
| Top-k 20k shaping | 20000 | feasible-topk, k=12 | 0.5 / 0.5 / 0.1 | 1.50 | 8/12 | -3.080 | -0.212 | 0.9292 | 29.4 | 1.7 | 0 |
| Top-k 5k shaping | 5000 | feasible-topk, k=12 | 0.5 / 0.5 / 0.1 | 2.67 | 4/12 | -7.540 | -4.228 | 0.8813 | 34.7 | 9.9 | 0 |
| Legal 20k shaping | 20000 | legal | 0.5 / 0.5 / 0.1 | 4.33 | 0/12 | -16.189 | -12.192 | 0.8309 | 52.8 | 19.4 | 0 |

当前最强 no-shaping 版本的场景级结果：

| Scenario | Best policy | Proposed-RL mean QoE | Proposed-RL mean rank | Gap to best |
|---|---|---:|---:|---:|
| compute_congested | Proposed-RL | -9.313 | 1.00 | 0.000 |
| default | Proposed-RL | -0.247 | 1.00 | 0.000 |
| low_snr | Proposed-RL | -1.055 | 1.00 | 0.000 |
| tight_deadline | Proposed-RL | -0.767 | 1.00 | 0.000 |

与强 deterministic baseline 的对比：

| Policy | Mean rank | Mean QoE | Gap to best | Success | Delay ms | Energy J | AoSI cost | Quality violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed-RL, top-k no-shaping | 1.00 | -2.845 | 0.000 | 0.9196 | 50.12 | 1.619 | 118.43 | 27.5 |
| Semantic-Greedy | 3.50 | -6.150 | -3.305 | 0.9282 | 75.93 | 1.203 | 116.68 | 29.0 |
| No-AoSI | 2.75 | -8.654 | -5.809 | 0.9155 | 90.46 | 1.371 | 116.98 | 28.0 |

## 7. Ablation 解释

### 7.1 图像质量特征是必要改进

metadata-only MLP 的 test R2 为 `0.4607`，新 image-feature MLP 为 `0.6357`。这说明原先 proxy 偏弱的关键原因不是 MLP 形式本身，而是输入只包含表格元信息，缺少亮度、清晰度、边缘、熵等 ROI 图像质量线索。

### 7.2 20k timesteps 明显优于 5k

同样使用 `feasible-topk + shaping` 时，5k formal 的 mean success rate 只有 `0.8813`，mean rank 为 `2.67`，mean QoE gap 为 `-4.228`。20k 后 success rate 提升到 `0.9292`，mean rank 到 `1.50`，gap 到 `-0.212`。之前 5k pilot 的失败主要来自训练不足和 seed 敏感，而不是 proxy 或候选排序彻底失效。

### 7.3 feasible-topk 是关键

`legal` 20k 在相同 shaping 和训练预算下 mean rank 只有 `4.33`，QoE gap 为 `-12.192`，success rate 为 `0.8309`。这说明当前 PPO 在完整 legal 动作空间里仍然难以稳定学习，候选剪枝对降低探索难度非常关键。

### 7.4 当前 shaping 权重未带来 QoE 收益

在 `feasible-topk + 20k` 下，no-shaping 的 mean QoE 为 `-2.845`，优于 shaping 的 `-3.080`；no-shaping 的 best count 为 `12/12`，shaping 为 `8/12`。不过 shaping 的 mean success rate 更高，`0.9292` 对 `0.9196`，因此二者取舍取决于下一阶段的主指标：

- 若以当前 QoE 和 mean success gate 为准，推荐 no-shaping 作为主配置。
- 若强调每个 seed-scenario 都要有高 success rate，需要继续保留 shaping 对照，或收紧 checkpoint selection 的 success 约束。

## 8. 门槛判断

| 门槛 | Top-k 20k no-shaping | Top-k 20k shaping | 结论 |
|---|---:|---:|---|
| Proposed-RL mean rank <= 2.5 | 1.00 | 1.50 | 均通过 |
| Proposed-RL mean success rate >= 0.90 | 0.9196 | 0.9292 | 均通过 |
| Proposed-RL mean QoE gap to best >= -2.0 | 0.000 | -0.212 | 均通过 |
| executed_illegal_count == 0 | 0 | 0 | 均通过 |
| quality violation 不明显高于强基线 | 27.5 | 29.4 | no-shaping 更稳，shaping 可接受 |

当前结果支持进入下一步 larger formal，但不建议继续使用 5k 作为结论性训练预算，也不建议把 `legal` 动作空间结果直接作为主结果。

## 9. 风险与限制

- 本轮仍使用 synthetic visibility/SNR traces，未接入 STK/TLE/ns-3。
- 当前 formal 是 512 ROI、3 seeds，尚不是论文级全量实验。
- profiling 使用 `test_full_image_features` 做工程验证；论文级结果应切换到严格 train/val/test 隔离的 profiling。
- no-shaping 虽然聚合 QoE 最好，但存在个别 seed-scenario success rate 低于 0.90 的情况；需要在更大规模实验中确认不是偶然收益。
- proxy 的 top-1 action agreement 仍只有 `0.1692`，说明 proxy 不是动作排序的完美 oracle；调度侧仍需要依赖候选约束和 RL 学习，而不能只看 R2。

## 10. 下一步建议

1. 将下一轮 larger formal 的主配置改为 `feasible-topk + k=12 + 20k + no-shaping`，同时保留 `feasible-topk + 20k + shaping` 作为成功率稳健性对照。
2. 把 checkpoint selection 从当前 mean success gate 扩展为 seed-scenario 级别 gate，例如要求 selected checkpoint 的 success rate 不低于 0.90，或对 drop_count 增加上限。
3. 扩大 ROI 数和 seeds 后复查 no-shaping 的优势是否稳定。
4. 在论文级实验前，重新生成严格隔离的 profiling 和 proxy，避免把工程验证结果误写成最终泛化结论。
5. 如果 larger formal 中 no-shaping 仍稳定领先，再把当前 shaping 权重从主线移到 ablation；如果 success rate 波动扩大，则优先优化 checkpoint selection，而不是盲目调大 proxy。
