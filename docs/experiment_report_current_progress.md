# LEO 遥感图像语义通信项目当前进展与下一步执行计划

报告日期：2026-06-22

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

当前还不能把 PPO 结果作为论文最终正向结论。原因是：主论文一致的 `legal` candidate mode 已完成一次 3-seed formal run，但 Proposed-RL 仍未稳定优于强基线；质量代理测试集 `R2` 仍偏低；链路仍是可复现 synthetic trace，尚未接入 STK/TLE/ns-3 或真实轨道链路。

本轮梳理和文档更新不改变论文公式，仅校正工程进展记录和下一步执行顺序。

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

- 原文预期最终应有稳健的 Proposed-RL/MADRL 对比结论；当前只有单智能体 PPO pilot 和候选动作消融。
- 原文提到更高保真在轨模拟；当前是 synthetic visibility/SNR trace。
- 原文强调代理要可靠防止“过度压缩/过早退出作弊”；当前 MLP proxy 已接入，但精度还需要提高。

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

Profiling 已用 test GT ROI crops、formal multi-exit checkpoint 和 `local + beta_0..3` 生成：

| 项 | 数值 |
|---|---:|
| ROI 数 | 6842 |
| Exit 数 | 4 |
| 压缩/本地等级 | 5 |
| profiling 行数 | 136840 |

质量代理当前输入排除了泄漏字段 `quality_label`、`task_quality`、`predicted_class_id`、`exit_confidence`。当前 `outputs/proxy/test_full/quality_proxy.joblib` 指标：

| Split | MAE | MSE | R2 |
|---|---:|---:|---:|
| train | 0.1319 | 0.0380 | 0.6692 |
| test | 0.1893 | 0.0687 | 0.4607 |

结论：质量代理链路已经打通，但测试集解释力仍偏弱，应作为下一步优先优化对象。

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
| legal candidate formal | 3 seeds, 512 ROI, 5000 timesteps | 合法动作执行稳定，但仍弱于 Semantic-Greedy/No-AoSI |
| feasible-topk candidate pilot | 2 seeds, 128 ROI, 2048 timesteps | Proposed-RL 在 pilot 中最好，但属于候选生成消融 |
| feasible-topk formal | 3 seeds, 512 ROI, 5000 timesteps | Proposed-RL 在消融设置下整体最好 |

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

解释：`legal` 是更接近论文主方法的物理合法动作候选；`feasible-topk` 可以作为候选动作剪枝/生成消融，不应静默替代主方法。

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

结论：`legal` candidate mode 已证明不会执行物理非法卸载，但当前 PPO 训练目标和动作选择仍偏保守/不稳定。下一步不应把这个结果包装成主方法优势，而应先做 reward/observation/proxy 改进，或把 `feasible-topk` 作为候选生成消融单独报告。

`feasible-topk` formal 消融也已于 2026-06-22 完成，配置同为 3 seeds、512 ROI、5000 timesteps/scenario、4 个 stress scenarios，区别是每步只保留 `drop + top-12` 个合法非丢弃候选动作。聚合结果如下：

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

解释：top-k 候选剪枝显著改善了 PPO 可学习性，说明当前瓶颈很可能来自原始合法候选动作空间过宽和动作索引学习困难，而不只是奖励公式本身。不过该设置引入了启发式候选生成，论文表述必须标为 ablation/candidate-generation variant。

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

### P1：提高质量代理可信度

1. 生成 train/val/test 分离的 profiling 表，而不是主要依赖 `test_full` 训练 proxy。
   目的：让 proxy 训练、模型选择和最终调度评估边界更清楚。

2. 训练更稳健的 proxy baseline。
   可尝试：更强 MLP、RandomForest/GradientBoosting 回归器、按类别/出口分组误差校准。

3. 质量门槛：优先把 held-out `R2` 从当前 `0.4607` 提升到更可靠区间，同时观察 high-value ROI 分组误差。

训练预估：单次 sklearn proxy 训练通常约 1-5 分钟；若重新生成大 profiling 表，取决于 GPU 和 ROI 数，预计 10-60 分钟。

### P2：完成 paper-consistent PPO formal run

主线命令建议：

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --candidate-mode legal --seeds 42 43 44 --limit-rois 512 --total-timesteps 5000 --eval-frequency 500 --checkpoint-frequency 1000 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --output-dir outputs\rl\ppo_legal_formal --exist-ok
```

状态：已完成一次 `outputs\rl\ppo_legal_formal` run。启动前保守预估为 24-96 分钟，当前开发机实际约 3-4 分钟。

验收重点：

- `Proposed-RL` 没有在 `legal` candidate mode 下稳定接近或超过 `Semantic-Greedy`、`No-AoSI`。
- 每个 scenario 已有 PPO rank、gap、success、timeout、quality violation 聚合报告。
- `executed_illegal_count` 为 0，物理合法动作约束生效。
- 后续应重点排查 reward shaping、candidate remap、proxy 误差和 AoSI 代价权重。

### P3：保留 feasible-topk 作为消融

当前 `feasible-topk` pilot 表现最好，但它把动作空间预先裁成 top-k 合法动作，因此应作为候选生成消融而不是默认主方法。

正式消融命令建议：

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --candidate-mode feasible-topk --candidate-top-k 12 --seeds 42 43 44 --limit-rois 512 --total-timesteps 5000 --eval-frequency 500 --checkpoint-frequency 1000 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --output-dir outputs\rl\ppo_topk_formal --exist-ok
```

状态：已完成一次 `outputs\rl\ppo_topk_formal` run。结果显示 Proposed-RL 在候选剪枝消融中整体最好，但 `low_snr` 场景仍输给 AoSI-Greedy。

### P4：视觉前端和高保真链路增强

1. 检测器校准：针对原图级高 FP，调整 confidence/NMS 或做类别校准。
   目标：不显著损失高价值类别 recall 的前提下降低 ROI 流量。

2. 高保真链路：接入 STK/TLE/ns-3 或真实轨道链路表。
   当前 synthetic trace 足以验证因果闭环，但不等价于在轨物理实验。

3. 论文图表：在 formal run 完成后生成 QoE、success、delay、energy、AoSI、traffic、CDF、training curve 和场景热力图。

## 6. 立即执行记录

本轮已经开始执行 P0：

- 重新梳理并更新当前阶段报告。
- 新增 `requirements.txt` 和 `environment.yml`，固化当前 `leo_semantic` 环境的核心运行依赖。
- 明确 `legal` 是主论文一致候选动作设置，`feasible-topk` 是消融设置。
- 把后续训练命令和训练时间预估写入报告。
- 已执行 `ppo_legal_formal` dry-run，确认 3 seeds x 4 scenarios 的训练命令能正确构造；预计耗时约 24-96 分钟。
- 已完成 `ppo_legal_formal` 正式运行并写入聚合报告；结果显示 Proposed-RL 在合法候选主设置下仍落后于强启发式基线。

下一步建议回到 `legal` 主设置改进 reward、候选动作表达和 proxy。top-k 结果可作为候选生成消融支持“动作空间剪枝有助于 PPO 学习”的论点，但不能静默替代主方法。

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
