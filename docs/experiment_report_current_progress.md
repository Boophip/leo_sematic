# LEO 遥感图像语义通信项目当前进展与下一步执行计划

报告日期：2026-06-24

## 1. 当前判断

项目已经从早期“可信视觉数据”阶段推进到可运行的合成 LEO 链路调度闭环。现有代码已经覆盖：

```text
DOTA 原图级划分
    -> 重叠切片与 YOLO-OBB 数据转换
    -> 轻量 ROI 检测器训练和原图级恢复/NMS
    -> ROI 元数据、裁剪、语义价值和 AoSI 网格输入
    -> GT ROI crops
    -> 多出口任务模型
    -> ROI x exit x compression profiling
    -> MLP 质量代理
    -> SNR-AMC、队列、能耗、AoSI 的确定性仿真
    -> 单智能体 PPO smoke/pilot 与多场景聚合报告
```

当前已经完成 strict train/val/test 隔离质量代理，以及 strict proxy 下 `feasible-topk(k=12)` PPO 的 20k/50k/100k formal run。方法边界已经调整：`feasible-topk` 纳入论文主方法，作为 Proxy-Guided Feasible Top-k PPO 的候选动作生成模块；`legal` 全动作空间 PPO 保留为消融/压力对照。

当前仍不能过度表述的部分是：链路仍是可复现 synthetic trace，尚未接入 STK/TLE/ns-3 或真实轨道链路；当前实现仍是单智能体 PPO，不能声称已经完成 MADRL。

本轮梳理和文档更新不改变论文公式、QoE 定义、AoSI 定义或动作语义，仅明确主方法的候选动作生成机制。

## 2. 与 `project.pdf` 原文思路对照

当前实现与原文一致的关键点：

- 源卫星侧使用轻量 ROI 检测器，避免传输完整大图。
- 删除模型中间切分，只保留本地早退和跨星完整卸载。
- 本地动作不使用通信压缩，跨星卸载才遍历压缩等级。
- 在线调度使用离线 profiling 和 MLP 质量代理，不在 RL step 内运行视觉模型。
- 星间通信采用固定发射功率和 SNR-AMC 查表，不做每步连续功率凸优化。
- 计算队列按固定频率演化，能耗按固定频率/固定发射功率建模。
- AoSI 使用静态空间网格、概率 OR 网格价值和 soft reset，避免目标级跨帧跟踪假设。
- 不可见链路作为非法动作处理，而不是用普通大时延代替。

当前与原文目标仍有差距的部分：

- 原文预期最终应有稳健的 Proposed-RL/MADRL 对比结论；当前是单智能体 Proxy-Guided Feasible Top-k PPO，还未升级为 MADRL。
- 原文提到更高保真在轨模拟；当前是 synthetic visibility/SNR trace。
- 原文强调代理要可靠防止“过度压缩/过早退出作弊”；当前 strict image-feature MLP proxy 已显著改善，但仍需在论文中说明 top-1 action agreement 并非 oracle 级别。

## 3. 已完成阶段与主要结果

### 3.1 DOTA 数据与 YOLO-OBB 前端

当前使用 DOTA-v1.0 可用训练图像构建 500 张原图子集，随机种子 42：

| Split | 原图数 | 原始目标数 |
|---|---:|---:|
| train | 300 | 20741 |
| val | 100 | 6922 |
| test | 100 | 7003 |

切片配置为 `tile_size=1024`、`overlap=200`、`min_visible_ratio=0.7`：

| Split | 切片数 | 切片内目标数 | 空标签切片 |
|---|---:|---:|---:|
| train | 3926 | 38835 | 1615 |
| val | 1295 | 12525 | 551 |
| test | 1310 | 13912 | 580 |

YOLO11n-OBB test 切片级指标：

| 指标 | 数值 |
|---|---:|
| Precision | 0.7655 |
| Recall | 0.7037 |
| mAP50 | 0.7209 |
| mAP50-95 | 0.5450 |
| 参数量 | 2.66M |
| FLOPs | 16.83G |
| 单切片推理 | 6.27 ms |

原图级恢复/NMS 后 test 指标：

| 指标 | 数值 |
|---|---:|
| Precision | 0.3334 |
| Recall | 0.9284 |
| mAP50 | 0.6911 |
| TP | 6352 |
| FP | 12700 |

结论：检测前端已经达到高召回阶段目标，但假阳性仍偏多，会放大后续调度负载。

### 3.2 ROI 元数据、标签与 GT crops

Detector ROI metadata 已生成：

| 项 | 数值 |
|---|---:|
| test 原图数 | 100 |
| detector ROI 数 | 19185 |
| ROI 裁剪图 | 19185 |
| AoSI 网格单元 | 6400 |
| 非空网格单元 | 3252 |

ROI 与 DOTA GT 匹配标签：

| 项 | 数值 |
|---|---:|
| ROI 总数 | 19185 |
| GT 数 | 6842 |
| true_positive | 6352 |
| false_positive | 12700 |
| class_mismatch | 310 |
| background_fp | 12390 |
| ignored_prediction | 133 |

GT ROI crops 已按原图 split 隔离生成，用于监督多出口模型：

| Split | 图像数 | ROI crops |
|---|---:|---:|
| train | 299 | 19336 |
| val | 100 | 6656 |
| test | 100 | 6842 |

### 3.3 多出口模型、Profiling 与质量代理

多出口模型 formal baseline 已完成 20 epoch 训练。最佳 epoch 为 19，val mean accuracy 为 `0.7188`：

| Exit | Val Accuracy |
|---|---:|
| 1 | 0.4085 |
| 2 | 0.7267 |
| 3 | 0.8529 |
| 4 | 0.8872 |

Profiling 已用 GT ROI crops、formal multi-exit checkpoint 和 `local + beta_0..3` 生成，并已经切换到严格 image-level train/val/test 隔离版本：

| 项 | 数值 |
|---|---:|
| train profiling 行数 | 386720 |
| val profiling 行数 | 133120 |
| test profiling 行数 | 136840 |
| Exit 数 | 4 |
| 压缩/本地等级 | 5 |

质量代理当前输入排除了泄漏字段 `quality_label`、`task_quality`、`predicted_class_id`、`exit_confidence`。当前 `outputs/proxy/strict_image_features/quality_proxy.joblib` 指标：

| Split | MAE | MSE | R2 |
|---|---:|---:|---:|
| train | 0.0554 | 0.0097 | 0.9141 |
| val | 0.0905 | 0.0253 | 0.7822 |
| test | 0.0850 | 0.0234 | 0.8023 |

结论：质量代理链路已经从 metadata-only baseline 升级到 strict image-feature MLP。测试集解释力已达到当前论文级结果整理门槛，但 action top-1 agreement 仍不是 oracle 级别，因此候选生成和 PPO 学习仍需要共同发挥作用。

### 3.4 确定性卫星仿真

`src/simulation/` 和 `src/envs/` 已实现并测试：

- SNR-AMC 查表和通信时延。
- 不可见链路非法动作。
- 固定频率计算队列。
- 计算能耗和通信能耗。
- 网格 AoSI 价值、变化因子和 soft reset。
- Random、Local-Deep、Local-Adaptive、Best-SNR、Semantic-Greedy、No-AoSI、AoSI-Greedy 基线。

确定性四场景中，当前强基线表现如下：

| Scenario | 当前最好基线 | QoE |
|---|---|---:|
| default | No-AoSI | -12.4728 |
| low_snr | No-AoSI | -17.1291 |
| compute_congested | Semantic-Greedy | -35.9335 |
| tight_deadline | No-AoSI | -26.8795 |

### 3.5 PPO Pilot 与候选动作实验

已有 PPO smoke/pilot 和 formal experiment manager。当前主要 pilot 结果：

| 设置 | 规模 | Proposed-RL 状态 |
|---|---|---|
| fixed action pilot | 2 seeds, 64 ROI, 128 timesteps | 明显弱于强基线 |
| legal candidate pilot | 2 seeds, 128 ROI, 2048 timesteps | 合法动作执行稳定，但均值仍弱于 Semantic-Greedy |
| legal candidate formal | 3 seeds, 512 ROI, 5000 timesteps | 作为 full action-space 对照，合法动作执行稳定，但仍弱于 Semantic-Greedy/No-AoSI |
| feasible-topk candidate pilot | 2 seeds, 128 ROI, 2048 timesteps | Proposed-RL 在 pilot 中最好，推动其升级为主方法候选生成机制 |
| feasible-topk formal | 3 seeds, 512 ROI, 5000 timesteps | 早期 top-k formal 整体最好，但仍使用旧 proxy/短训练预算 |

`legal` candidate pilot 结果：

| Policy | Mean QoE | Mean Success | Mean Delay ms |
|---|---:|---:|---:|
| Semantic-Greedy | -6.9979 | 0.8691 | 268.37 |
| Proposed-RL | -8.2969 | 0.7852 | 213.25 |
| No-AoSI | -9.3731 | 0.8105 | 311.56 |

`feasible-topk` candidate pilot 结果：

| Policy | Mean QoE | Mean Success | Mean Delay ms |
|---|---:|---:|---:|
| Proposed-RL | -2.2353 | 0.9209 | 100.48 |
| Semantic-Greedy | -6.9979 | 0.8691 | 268.37 |
| No-AoSI | -9.3731 | 0.8105 | 311.56 |

解释：这一轮历史结果说明，完整合法动作空间虽然能保证不执行不可见链路卸载，但 PPO 学习难度过高；`feasible-topk` 显著降低动作空间复杂度，因此后续将其明确纳入主方法，而不是静默替代 full legal PPO。

`legal` candidate formal run 已于 2026-06-22 完成，配置为 3 seeds、512 ROI、5000 timesteps/scenario、4 个 stress scenarios。脚本启动前给出的保守预估为 24-96 分钟，当前开发机实际约 3-4 分钟完成。聚合结果如下：

| Policy | Mean QoE | Mean Rank | Best Count | Mean Success | Mean Delay ms | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-Greedy | -7.8023 | 2.9167 | 0 | 0.9360 | 98.82 | 0 |
| No-AoSI | -10.4192 | 2.4167 | 3 | 0.9214 | 112.85 | 0 |
| Proposed-RL | -14.3282 | 4.3333 | 0 | 0.8449 | 83.49 | 0 |
| AoSI-Greedy | -16.0707 | 2.7500 | 6 | 0.8965 | 150.86 | 0 |

按场景看，Proposed-RL 对当前最好策略仍有差距：

| Scenario | Best Policy | Best Mean QoE | PPO Mean QoE | PPO Mean Gap |
|---|---|---:|---:|---:|
| default | AoSI-Greedy | -2.9404 | -7.0047 | -4.0643 |
| low_snr | AoSI-Greedy | -2.2056 | -8.7541 | -6.5485 |
| compute_congested | Local-Adaptive | -13.5811 | -29.1026 | -15.5214 |
| tight_deadline | No-AoSI | -1.6467 | -12.4516 | -10.8049 |

结论：`legal` candidate mode 已证明不会执行物理非法卸载，但 PPO 在 full legal action-space 下仍偏保守/不稳定。该结果保留为主方法的压力对照，用于支撑候选动作生成的必要性。

`feasible-topk` 早期 formal 已于 2026-06-22 完成，配置同为 3 seeds、512 ROI、5000 timesteps/scenario、4 个 stress scenarios，区别是每步只保留 `drop + top-12` 个合法非丢弃候选动作。聚合结果如下：

| Policy | Mean QoE | Mean Rank | Best Count | Mean Success | Mean Delay ms | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|
| Proposed-RL | -4.2465 | 1.9167 | 7 | 0.9258 | 53.18 | 0 |
| Semantic-Greedy | -7.8023 | 3.5833 | 0 | 0.9360 | 98.82 | 0 |
| No-AoSI | -10.4192 | 3.0000 | 1 | 0.9214 | 112.85 | 0 |
| AoSI-Greedy | -16.0707 | 3.0833 | 4 | 0.8965 | 150.86 | 0 |

按场景看：

| Scenario | Best Policy | Best Mean QoE | PPO Mean QoE | PPO Mean Gap |
|---|---|---:|---:|---:|
| default | Proposed-RL | -2.0709 | -2.0709 | 0.0000 |
| compute_congested | Proposed-RL | -7.3429 | -7.3429 | 0.0000 |
| low_snr | AoSI-Greedy | -2.2056 | -5.7212 | -3.5156 |
| tight_deadline | No-AoSI | -1.6467 | -1.8512 | -0.2045 |

解释：top-k 候选生成显著改善了 PPO 可学习性，说明当前瓶颈很可能来自原始合法候选动作空间过宽和动作索引学习困难，而不只是奖励公式本身。该机制已经升级为主方法的候选动作生成模块。

### 3.6 Strict Proxy 主方法 Formal 结果

6 月 24 日已完成 strict proxy + feasible-topk 的 20k/50k/100k formal run。所有 run 使用：

```text
profile: data/profiling/dota_v1_lite_300_100_100/test_strict_image_features/roi_profile_smoke.csv
proxy: outputs/proxy/strict_image_features/quality_proxy.joblib
candidate mode: feasible-topk
candidate top-k: 12
seeds: 42, 43, 44
scenarios: default, low_snr, compute_congested, tight_deadline
limit_rois: 512
```

当前主结果配置：

```text
strict proxy + feasible-topk(k=12) + shaping + 100k
```

聚合结果：

| Policy | Mean Rank | Best Count | Mean QoE | Mean Success | Mean Delay ms | Mean AoSI Cost | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed-RL | 1.0000 | 12 / 12 | -2.0784 | 0.9583 | 63.2113 | 115.4084 | 0.0000 |
| No-AoSI | 2.7500 | 0 / 12 | -10.9356 | 0.9092 | 108.9976 | 122.1974 | 0.0000 |
| Semantic-Greedy | 4.1667 | 0 / 12 | -16.3247 | 0.8608 | 113.4274 | 137.3001 | 0.0000 |

no-shaping 100k 对照也已完成，训练前脚本保守估计为 24-96 分钟，实际本机约 44.9 分钟。结果如下：

| Run | Mean Rank | Best Count | Mean QoE | Mean Gap | Mean Success | Mean Delay ms | Mean AoSI Cost | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k no-shaping | 1.0833 | 11 / 12 | -2.1354 | -0.0260 | 0.9478 | 62.8920 | 116.8893 | 0.0000 |
| 100k shaping | 1.0000 | 12 / 12 | -2.0784 | 0.0000 | 0.9583 | 63.2113 | 115.4084 | 0.0000 |

结论：no-shaping 100k 仍然明显优于强基线，但略弱于 shaping。当前论文主结果继续使用 `strict proxy + feasible-topk(k=12) + shaping + 100k`；no-shaping 100k 作为训练奖励消融。

场景级结果：

| Scenario | Best Policy | Proposed-RL QoE | Proposed-RL Rank | Gap |
|---|---|---:|---:|---:|
| compute_congested | Proposed-RL | -6.2276 | 1.0000 | 0.0000 |
| default | Proposed-RL | -0.6667 | 1.0000 | 0.0000 |
| low_snr | Proposed-RL | -1.0817 | 1.0000 | 0.0000 |
| tight_deadline | Proposed-RL | -0.3374 | 1.0000 | 0.0000 |

结论：Proxy-Guided Feasible Top-k PPO 已经给出当前可写入论文主结果的正向证据。full `legal` PPO 作为对照，说明候选动作生成不是装饰性技巧，而是星载大动作空间下提高学习效率的核心机制。

## 4. 当前验证状态

在项目专用环境中运行：

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

结果：

```text
Ran 87 tests in 4.541s
OK
```

注意：不要使用系统默认 Python 直接跑测试；默认环境缺少 `torch`、`cv2`、`gymnasium`、`stable_baselines3`、`joblib`、`sklearn` 等依赖。

## 5. 下一步工作优先级

### P0：把当前状态固化为可复现基线

1. 更新阶段报告和下一步计划。
   状态：本文件已更新。

2. 固化 Python/conda 环境依赖。
   状态：已新增 `requirements.txt` 和 `environment.yml`，避免测试在默认 Python 中失败。

3. 跑一遍完整测试。
   状态：已用 `F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v` 通过，`Ran 87 tests ... OK`。

### P1：固化 Proxy-Guided Feasible Top-k PPO 主方法

1. 将 `feasible-topk` 写入方法章节，作为主方法的候选动作生成模块。
   状态：已新增 `docs/feasible_topk_main_method.md`。

2. 明确候选生成规则：先过滤物理非法动作，再按 proxy 质量、时延和通信字节数保留 top-k。
   状态：已在方法说明与 pipeline 文档中固定。

3. 保留 `legal` 全动作空间 PPO 作为消融/压力对照。
   目的：证明候选生成机制对 PPO 学习效率是必要的，而不是结果包装。

### P2：论文级结果图表与报告

1. 汇总 strict proxy、deterministic sanity、20k/50k/100k PPO 和 no-shaping 对照。
2. 生成和检查 QoE、success、delay、energy、AoSI、CDF、training curve 图表。
3. 用 `outputs/rl/ppo_topk_strict_proxy_shaping_100k` 作为当前主结果目录。

### P3：补跑 no-shaping 更大规模对照

为了确认 shaping/no-shaping 取舍，建议补跑 strict proxy + feasible-topk + no-shaping 的 100k formal：

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --candidate-mode feasible-topk --candidate-top-k 12 --seeds 42 43 44 --limit-rois 512 --total-timesteps 100000 --eval-frequency 2000 --checkpoint-frequency 20000 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --profile-csv data\profiling\dota_v1_lite_300_100_100\test_strict_image_features\roi_profile_smoke.csv --quality-proxy outputs\proxy\strict_image_features\quality_proxy.joblib --output-dir outputs\rl\ppo_topk_strict_proxy_no_shaping_100k --exist-ok
```

状态：已完成。训练前脚本保守估计约 24-96 分钟，实际本机约 44.9 分钟。

验收重点：

- 与 100k shaping 对比 mean QoE、mean success、quality violation 和 timeout。
- no-shaping 100k 的 mean rank 为 1.0833、best count 为 11/12、mean QoE 为 -2.1354，略弱于 shaping 100k。
- 当前论文主结果保留 shaping，no-shaping 作为训练奖励消融。

### P4：论文风险补强

1. strict split 防泄漏说明：proxy selection 只看 val，test 只做 final report。
2. synthetic trace 限制说明：当前结果不是 STK/TLE/ns-3 高保真链路。
3. 单智能体边界说明：当前不是 MADRL。
4. 检测器校准：针对原图级高 FP，调整 confidence/NMS 或做类别校准。
   目标：不显著损失高价值类别 recall 的前提下降低 ROI 流量。

5. 高保真链路：接入 STK/TLE/ns-3 或真实轨道链路表。
   当前 synthetic trace 足以验证因果闭环，但不等价于在轨物理实验。

## 6. 立即执行记录

本轮已经开始执行 P0：

- 重新梳理并更新当前阶段报告。
- 新增 `requirements.txt` 和 `environment.yml`，固化当前 `leo_semantic` 环境的核心运行依赖。
- 明确 `feasible-topk` 是主方法候选动作生成模块，`legal` 是 full action-space 消融/压力对照。
- 把后续训练命令和训练时间预估写入报告。
- 已执行 `ppo_legal_formal` dry-run，确认 3 seeds x 4 scenarios 的训练命令能正确构造；预计耗时约 24-96 分钟。
- 已完成 `ppo_legal_formal` 正式运行并写入聚合报告；结果显示 Proposed-RL 在 full legal action-space 下仍落后于强启发式基线。
- 已完成 strict proxy + feasible-topk 20k/50k/100k formal run；100k shaping 当前为主结果配置。
- 已完成 strict proxy + feasible-topk + no-shaping 100k 对照；结果支持把 no-shaping 放入 ablation。

下一步建议将主方法图表与论文实验口径整理到统一报告中。

## 7. 主要产物路径

| 产物 | 路径 |
|---|---|
| DOTA 子集摘要 | `data/DOTA-demo/summary.json` |
| 切片摘要 | `data/tiles/dota_demo_1024_o200/summary.json` |
| YOLO-OBB 数据集配置 | `data/yolo_obb/dota_demo_1024_o200/dataset.yaml` |
| 检测器权重 | `outputs/detector/yolo11n_obb_full/weights/best.pt` |
| 原图级评估报告 | `outputs/detector/yolo11n_obb_full_test_original/original_eval_report.json` |
| Detector ROI 元数据 | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl` |
| Detector ROI 标注 | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.jsonl` |
| GT ROI crops | `data/roi_crops_gt/dota_v1_lite_300_100_100/{train,val,test}/roi_metadata_labeled.jsonl` |
| 多出口 formal 权重 | `outputs/multi_exit/formal/best.pt` |
| Profiling 表 | `data/profiling/dota_v1_lite_300_100_100/test_full/roi_profile_smoke.csv` |
| 质量代理 | `outputs/proxy/test_full/quality_proxy.joblib` |
| 确定性仿真报告 | `outputs/simulation/deterministic/comparison_report.md` |
| PPO legal pilot | `outputs/rl/ppo_candidate_legal_pilot/comparison_report.md` |
| PPO top-k pilot | `outputs/rl/ppo_candidate_topk_pilot/comparison_report.md` |
| PPO legal formal | `outputs/rl/ppo_legal_formal/comparison_report.md` |
| PPO top-k formal | `outputs/rl/ppo_topk_formal/comparison_report.md` |
| PPO strict top-k 100k 主结果 | `outputs/rl/ppo_topk_strict_proxy_shaping_100k/comparison_report.md` |
| PPO strict top-k 100k no-shaping 对照 | `outputs/rl/ppo_topk_strict_proxy_no_shaping_100k/comparison_report.md` |
| feasible-topk 主方法说明 | `docs/feasible_topk_main_method.md` |
