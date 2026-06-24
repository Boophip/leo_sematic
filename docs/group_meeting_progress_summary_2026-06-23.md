# LEO 遥感图像语义通信项目组会总结

报告日期：2026-06-23

本文用于组会汇报，目标是简洁、准确地说明当前工作进展、实验流程、未完成部分，以及当前实现与 `project.pdf` 原文思路的对应关系。本文只总结已有代码、文档和输出结果，不修改研究公式、QoE、AoSI 或 PPO 环境定义。

## 1. 一句话结论

项目已经完成从 DOTA 遥感图像到 ROI 提取、质量建模、LEO 调度环境和 PPO 训练评估的可运行闭环。当前最新进展是：在质量代理中加入 ROI 图像质量特征后，proxy 精度明显提升；在 `feasible-topk + 20k timesteps` 设置下，Proposed-RL 已能稳定超过当前强基线。

但这还不是论文最终结论。主要原因是：当前最好的 PPO 结果依赖 `feasible-topk` 候选动作剪枝；完整 `legal` 动作空间下 PPO 仍不稳定；链路仍是 synthetic visibility/SNR trace；profiling 和 proxy 还需要切换到严格 train/val/test 隔离后再做论文级实验。

可以在组会上这样概括：

```text
当前已经跑通了完整实验闭环，并且新轻量质量代理和 top-k PPO ablation 给出了积极结果。
但是主方法还需要明确：top-k 是正式方法的一部分，还是只作为候选生成消融。
下一步重点是扩大 formal 规模、严格数据隔离、稳定 checkpoint 选择，并补齐高保真链路和论文图表。
```

## 2. 当前实验流程

当前完整流程如下：

```text
DOTA 原图
  -> 原图级 train/val/test 划分
  -> 重叠切片和 YOLO-OBB 数据转换
  -> YOLO11n-OBB 训练与测试
  -> 切片检测结果恢复到原图坐标
  -> 全局旋转框 NMS 去重
  -> 生成 detector ROI metadata
  -> 计算 ROI 语义价值和 AoSI 网格编号
  -> detector ROI 与 DOTA GT 匹配，得到 TP/FP 标签
  -> 构建 GT ROI crops
  -> 训练多出口任务模型
  -> 遍历 ROI x exit x compression，生成离线 profiling 表
  -> 训练质量代理 proxy
  -> 构建 LEO 链路、计算队列、能耗、AoSI 调度环境
  -> 运行确定性基线
  -> 训练和评估 PPO 调度策略
  -> 输出 formal 聚合报告和图表
```

这个流程分成两个阶段：

| 阶段 | 作用 | 当前状态 |
|---|---|---|
| 离线视觉建模 | 生成 ROI、动作质量和动作代价的可查表数据 | 已打通 |
| 在线调度仿真 | 根据链路、队列、质量和 AoSI 选择处理动作 | 已打通，仍需扩大 formal |

当前在线环境不直接运行 YOLO 或多出口 CNN，只调用 profiling 表、质量代理和仿真模型。这一点与原文“在轨极速映射”和“避免每步运行高开销视觉模型”的思路一致。

## 3. 已完成工作

### 3.1 DOTA 数据划分与切片

已基于 DOTA-v1.0 构建 500 张原图子集，随机种子为 42。划分在原图级完成，避免同一原图产生的切片或 ROI 跨 train/val/test 泄漏。

| Split | 原图数 | 原始目标数 |
|---|---:|---:|
| train | 300 | 20741 |
| val | 100 | 6922 |
| test | 100 | 7003 |

切片配置为 `tile_size=1024`、`overlap=200`、`min_visible_ratio=0.7`。

| Split | 切片数 | 切片内目标数 | 空标签切片 |
|---|---:|---:|---:|
| train | 3926 | 38835 | 1615 |
| val | 1295 | 12525 | 551 |
| test | 1310 | 13912 | 580 |

这一步解决的是数据可信度问题：先按原始大图划分，再切片，避免测试集和训练集出现近邻区域泄漏。

### 3.2 YOLO11n-OBB ROI 检测前端

已完成 YOLO11n-OBB formal 训练、切片级评估和原图级恢复/NMS 评估。

切片级 test 指标：

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
| NMS 后 ROI | 19185 |

结论：检测前端已经满足“高召回”的阶段目标。原图级 recall 达到 `0.9284`，说明关键目标大多能进入调度流程。但 precision 只有 `0.3334`，FP 较多，会增加后续计算、通信和队列负担。

### 3.3 ROI 元数据、语义价值与 AoSI 网格

已生成 detector ROI metadata，并将 ROI 恢复到原图坐标后分配到固定 AoSI 网格。每条 ROI 记录包含：

```text
image_id
global_obb
predicted_class
confidence
area_ratio
center_xy
grid_id
difficult
crop_path
semantic_value
```

当前 test split 结果：

| 项 | 数值 |
|---|---:|
| test 原图数 | 100 |
| detector ROI 数 | 19185 |
| ROI 裁剪图 | 19185 |
| AoSI 网格单元 | 6400 |
| 非空网格单元 | 3252 |

ROI 与 DOTA GT 匹配标签如下：

| 项 | 数值 |
|---|---:|
| ROI 总数 | 19185 |
| GT 数 | 6842 |
| true_positive | 6352 |
| false_positive | 12700 |
| class_mismatch | 310 |
| background_fp | 12390 |
| ignored_prediction | 133 |

ROI 语义价值使用类别优先级、检测置信度和面积比例加权：

```text
v_k = clip(
    w_class * class_priority(y_k)
    + w_confidence * confidence_k
    + w_area * area_ratio_k,
    0,
    1
)
```

当前工程权重为 `w_class=0.5`、`w_confidence=0.35`、`w_area=0.15`。语义价值只表示 ROI 的业务重要性，不表示处理是否正确；处理质量由后续 proxy 预测。

### 3.4 GT ROI crops 与多出口任务模型

已构建 GT ROI crops，用于训练多出口任务模型。当前使用 GT crops 是为了先获得干净监督信号，避免 detector FP 直接污染任务模型。

| Split | 图像数 | GT ROI crops |
|---|---:|---:|
| train | 299 | 19336 |
| val | 100 | 6656 |
| test | 100 | 6842 |

多出口模型 formal baseline 已训练 20 epoch，最佳 epoch 为 19，val mean accuracy 为 `0.7188`。

| Exit | Val Accuracy |
|---|---:|
| 1 | 0.4085 |
| 2 | 0.7267 |
| 3 | 0.8529 |
| 4 | 0.8872 |

这说明 early-exit 维度已经具备真实 trade-off：浅层出口计算便宜但质量低，深层出口质量高但计算成本更高。

### 3.5 压缩策略与离线 Profiling

已实现本地处理和四级跨星卸载压缩策略。

| Level | 编码方式 | 缩放 | 用途 |
|---|---|---:|---|
| `local` | none | 1.00x | 本地处理，不产生通信 payload |
| `beta_0` | PNG lossless | 1.00x | 无损卸载质量上界 |
| `beta_1` | JPEG quality 90 | 1.00x | 高质量卸载 |
| `beta_2` | JPEG quality 70 | 0.75x | 中等压缩 |
| `beta_3` | JPEG quality 45 | 0.50x | 强压缩 |

已用 test GT ROI crops、formal multi-exit checkpoint 和 `local + beta_0..3` 生成 profiling 表。

| 项 | 数值 |
|---|---:|
| ROI 数 | 6842 |
| Exit 数 | 4 |
| 压缩/本地等级 | 5 |
| profiling 行数 | 136840 |

行数对应：

```text
6842 ROI x 4 exits x 5 compression/local levels = 136840 rows
```

profiling 记录每个动作组合下的压缩字节数、编解码时延、推理时延和真实任务质量，是在线调度环境的基础数据来源。

### 3.6 质量代理 Proxy

旧版 metadata-only MLP 已经打通，但测试泛化偏弱：test MAE 为 `0.1893`，test R2 为 `0.4607`。

6 月 23 日完成了新一轮轻量质量代理实验。新 proxy 在元信息基础上加入 ROI 图像质量特征，包括亮度、颜色统计、Laplacian 清晰度、边缘密度和熵等。

| Proxy 版本 | 输入特征 | 模型大小 | Test MAE | Test R2 | High-value MAE | Threshold F1 | Mean regret |
|---|---|---:|---:|---:|---:|---:|---:|
| Metadata-only MLP | 表格元信息 | 0.06 MB | 0.1893 | 0.4607 | - | - | - |
| Image-feature MLP 128/64/32 | 元信息 + 图像质量特征 | 0.53 MB | 0.1344 | 0.6357 | 0.1333 | 0.8599 | 0.0691 |
| HistGBDT | 元信息 + 图像质量特征 | 1.87 MB | 0.1497 | 0.6157 | 0.1526 | 0.8450 | 0.0741 |
| ExtraTrees | 元信息 + 图像质量特征 | 857.56 MB | 0.1339 | 0.6600 | 0.1355 | 0.8630 | 0.0670 |

结论：图像质量特征带来主要收益。ExtraTrees 指标略高，但体积接近 858 MB，不符合星载轻量代理目标；当前默认选择 0.53 MB 的 image-feature MLP。

需要注意：新 proxy 的 top-1 action agreement 仍只有 `0.1692`，说明它不是完美动作排序 oracle。调度策略仍需要依赖候选约束和 RL 学习，而不能只看 R2。

### 3.7 确定性卫星仿真环境

已实现确定性仿真闭环，核心模块包括：

| 模块 | 作用 |
|---|---|
| `src/simulation/link.py` | SNR-AMC 查表、吞吐率、通信时延、通信能耗 |
| `src/simulation/compute.py` | 固定频率队列、等待时延、计算能耗 |
| `src/simulation/aosi.py` | 网格语义价值、变化因子、AoSI 更新 |
| `src/simulation/episode.py` | 策略动作评估、episode 运行、基线策略 |
| `src/envs/leo_scheduling_env.py` | Gymnasium 环境，供 PPO 训练使用 |

已实现的基线策略包括 Random、Local-Deep、Local-Adaptive、Best-SNR、Semantic-Greedy、No-AoSI 和 AoSI-Greedy。

使用新轻量 proxy 后，四场景 deterministic sanity 聚合结果如下：

| Policy | Mean QoE | Success rate | Quality violation | Timeout | Illegal |
|---|---:|---:|---:|---:|---:|
| Semantic-Greedy | -6.150 | 0.9282 | 29.0 | 8.0 | 0 |
| No-AoSI | -8.654 | 0.9155 | 28.0 | 16.0 | 0 |
| AoSI-Greedy | -17.414 | 0.8804 | 28.0 | 35.3 | 0 |
| Local-Adaptive | -17.605 | 0.8613 | 41.0 | 32.0 | 0 |
| Best-SNR | -26.549 | 0.8291 | 52.0 | 38.5 | 0 |
| Local-Deep | -27.680 | 0.7886 | 52.0 | 62.5 | 0 |
| Random | -46.533 | 0.5708 | 169.0 | 34.3 | 0 |

该结果说明环境行为基本正常：强启发式基线明显优于 Random，且非法动作执行数为 0。

### 3.8 PPO 调度实验与 Ablation

已完成 PPO smoke、pilot、formal runner 和多场景聚合报告。当前正式实验使用 canonical QoE 评估策略，而不是直接用训练 reward 评估，避免 reward shaping 导致不可比。

6 月 22 日的初步 formal 显示：`legal` 5k 下 Proposed-RL 仍弱于强基线，`feasible-topk` 5k 有明显改善。

6 月 23 日进一步完成轻量 proxy 接入后的 20k formal 与 ablation。当前 Proposed-RL 聚合指标如下：

| 实验 | Timesteps | Candidate | Shaping 权重 | Mean rank | Best count | Mean QoE | Gap to best | Success | Illegal |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Top-k 20k no-shaping | 20000 | feasible-topk, k=12 | 0 / 0 / 0 | 1.00 | 12/12 | -2.845 | 0.000 | 0.9196 | 0 |
| Top-k 20k shaping | 20000 | feasible-topk, k=12 | 0.5 / 0.5 / 0.1 | 1.50 | 8/12 | -3.080 | -0.212 | 0.9292 | 0 |
| Top-k 5k shaping | 5000 | feasible-topk, k=12 | 0.5 / 0.5 / 0.1 | 2.67 | 4/12 | -7.540 | -4.228 | 0.8813 | 0 |
| Legal 20k shaping | 20000 | legal | 0.5 / 0.5 / 0.1 | 4.33 | 0/12 | -16.189 | -12.192 | 0.8309 | 0 |

当前最强配置是：

```text
profile: data/profiling/dota_v1_lite_300_100_100/test_full_image_features/roi_profile_smoke.csv
proxy: outputs/proxy/lightweight_image_features/quality_proxy.joblib
candidate_mode: feasible-topk
candidate_top_k: 12
total_timesteps: 20000
reward_shaping: 0.0 / 0.0 / 0.0
seeds: 42, 43, 44
```

该配置下，Proposed-RL 在 12 个 seed-scenario 样本中均为第一，mean rank 为 `1.00`，mean QoE gap to best 为 `0.000`，mean success rate 为 `0.9196`，executed illegal count 为 `0`。

但这必须谨慎表述：top-k 使用启发式候选剪枝，显著降低了动作空间难度。如果将其作为主方法，需要在方法章节明确说明；如果不作为主方法，则只能作为 candidate-generation ablation。

## 4. 当前实现与 `project.pdf` 原文思路对照

`project.pdf` 的核心思路是：不追求完整遥感图像无损传输，而是在链路时变、算力受限和语义信息易陈旧的条件下，优先让关键 ROI 被及时、正确地处理。

### 4.1 已经对齐的部分

| 原文思路 | 当前实现 | 状态 |
|---|---|---|
| 源卫星侧轻量 ROI 解析器 | YOLO11n-OBB 检测器 | 已实现 |
| ROI 语义价值连续建模 | 类别优先级、置信度、面积比例加权 | 已实现 |
| 使用 MLP 质量代理防止调度“作弊” | metadata-only MLP 与 image-feature MLP | 已实现，新版效果更好 |
| 去除模型中间切分 | 只保留本地早退和跨星完整卸载 | 已实现 |
| 本地 early-exit 与跨星卸载联合选择 | 动作包含节点、exit、compression | 已实现 |
| 固定功率 + SNR-AMC 查表 | `src/simulation/link.py` | 已实现 |
| 定频计算和队列演化 | `src/simulation/compute.py` | 已实现 |
| 静态网格化 AoSI | `src/simulation/aosi.py` | 已实现 |
| 不可见链路不能卸载 | 不可见链路为非法动作 | 已实现 |
| QoE 同时考虑质量、时延、能耗、AoSI | canonical QoE 与报告指标 | 已实现 |
| 质量和时延虚拟队列 | PPO 环境中已有相关 shaping/queue 机制 | 部分实现 |

### 4.2 与原文仍有差距的部分

| 原文目标 | 当前状态 | 差距 |
|---|---|---|
| 高保真在轨链路仿真 | synthetic visibility/SNR trace | 尚未接入 STK/TLE/ns-3 |
| 单层 MADRL | 当前为单智能体 PPO | MADRL 未开始 |
| 稳健的主方法优势 | top-k 20k 表现好，legal 20k 仍弱 | 主方法边界需明确 |
| 代理模型严格泛化验证 | 当前新 proxy 仍基于 `test_full_image_features` 工程验证 | 需严格 train/val/test profiling |
| 多数据集或更大规模测试 | 当前 DOTA lite 子集，512 ROI formal | 需扩大 ROI 数、seeds 和场景 |
| 论文级图表体系 | 已有部分图表输出 | 还需统一补齐 |

### 4.3 需要主动解释的工程设定

有几处设定与原文方向一致，但属于工程落地选择，组会上要主动说明：

- 当前压缩等级 `beta_0..3` 是可复现实验设定，原文没有规定具体 JPEG 参数。
- 当前 DOTA 使用 500 张原图子集，主要用于先打通流程和控制训练成本。
- 当前链路是 synthetic trace，能验证因果闭环，但不等价于高保真在轨实验。
- 当前 PPO 是单智能体版本，是 MADRL 前的必要验证步骤。
- 当前 `feasible-topk` 是候选动作剪枝。如果作为主方法，需要明确写入方法；如果不写入主方法，只能作为消融。

## 5. 目前还没有完成的部分

### 5.1 论文级数据隔离

当前新 proxy 使用 `test_full_image_features` 做工程验证。下一步需要生成严格 train/val/test 隔离的 profiling 和 proxy，避免把工程验证结果误写成最终泛化结论。

### 5.2 Proxy 还需要更稳健

新 image-feature MLP 已将 R2 从 `0.4607` 提升到 `0.6357`，但 top-1 action agreement 仍低，且不同类别、出口、压缩等级下误差可能不均衡。后续需要重点检查 high-value ROI 的误差。

### 5.3 `legal` 全动作空间 PPO 仍未稳定

`legal` 20k shaping 下 Proposed-RL mean rank 仍为 `4.33`，没有超过强基线。这说明完整合法动作空间仍过宽，raw action 到候选动作的映射学习困难。当前最好的 top-k 结果不能直接替代这一问题。

### 5.4 Formal 规模还不够

当前最强结果是 3 seeds、4 scenarios、512 ROI、20k timesteps。下一步需要扩大 ROI 数和 seeds，检查 no-shaping 的优势是否稳定，并提高 checkpoint selection 的逐 seed-scenario success gate。

### 5.5 检测器 FP 偏多

原图级 recall 高，但 precision 偏低，FP 会占用通信、计算和队列资源。后续需要调 detector confidence threshold、全局旋转 NMS 阈值，或做类别校准。

### 5.6 高保真链路未接入

当前 synthetic trace 足够验证环境因果关系，但还不是论文最终在轨链路实验。STK/TLE/ns-3 或真实轨道链路表尚未接入。

### 5.7 MADRL 未开始

当前是单智能体 PPO。只有在单智能体闭环、proxy 和动作空间都稳定后，才建议升级 MADRL。

### 5.8 论文图表还需统一生成

已有 formal 输出和部分图表，但论文级图表体系还需要补齐：

- QoE bar plot
- QoE CDF
- success rate
- delay mean/P95
- energy
- traffic bytes
- AoSI cost
- training curve
- scenario heatmap
- candidate mode ablation

## 6. 当前验证状态

本次复核使用项目 conda 环境运行：

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

结果：

```text
Ran 94 tests in 6.487s
OK
```

测试覆盖 DOTA 划分与切片、YOLO-OBB 转换与评估、ROI metadata、GT crops、多出口模型、压缩 profiling、质量代理防泄漏、SNR-AMC、计算队列、AoSI、Gymnasium 环境、PPO candidate mode 和 formal 聚合报告。

注意：不要使用系统默认 Python 直接跑测试；默认环境缺少 `torch`、`cv2`、`gymnasium`、`stable_baselines3`、`joblib`、`sklearn` 等依赖。

## 7. 下一步工作优先级

### P0：固化当前结果

保留当前 6 月 23 日轻量 proxy 和 PPO ablation 输出，避免后续调参覆盖。组会汇报时应明确：当前最强结果是 top-k 候选剪枝下的积极结果，不是 `legal` 全动作空间的最终胜利。

### P1：严格数据隔离并重训 proxy

生成 train/val/test 隔离的 profiling 表，再训练 image-feature MLP 和候选回归器。

训练预估：

- 单次 sklearn proxy 训练通常约 1 到 5 分钟。
- 重新生成大规模 profiling 表取决于 GPU 和 ROI 数，预计 10 到 60 分钟。

### P2：扩大 top-k formal 规模

以 `feasible-topk + k=12 + 20k + no-shaping` 作为下一轮主配置，同时保留 `feasible-topk + 20k + shaping` 作为成功率稳健性对照。

训练预估：

- 当前 20k formal 单次 3 seeds x 4 scenarios 实际约 8 到 9 分钟。
- 扩大 seeds 或 ROI 后耗时近似线性增加，建议训练前按新规模重新估计。

### P3：改进 checkpoint selection

当前 checkpoint gate 主要看 mean success。下一步建议改成 seed-scenario 级别 gate，例如要求每个 selected checkpoint 的 success rate 不低于 0.90，或限制 drop_count。

### P4：继续排查 `legal` PPO

若论文希望主方法不依赖 top-k，则必须继续优化 `legal` 设置。可尝试：

- 在 observation 中加入更直接的候选动作特征。
- 改进 raw action 到 legal candidate 的映射。
- 增加训练步数和 seeds。
- 比较 shaping/no-shaping 对 `legal` 的影响。

### P5：降低 detector FP

在不显著损失 high-value recall 的前提下，调整 confidence threshold、全局 NMS 或类别校准，降低无效 ROI 流量。

### P6：接入高保真链路与补图表

在主流程稳定后，接入 STK/TLE/ns-3 或真实链路表，并统一生成论文图表。

## 8. 主要产物路径

| 产物 | 路径 |
|---|---|
| 项目背景 | `PROJECT_CONTEXT.md` |
| 原文 PDF | `project.pdf` |
| DOTA 子集摘要 | `data/DOTA-demo/summary.json` |
| 切片摘要 | `data/tiles/dota_demo_1024_o200/summary.json` |
| YOLO-OBB 数据集配置 | `data/yolo_obb/dota_demo_1024_o200/dataset.yaml` |
| 检测器权重 | `outputs/detector/yolo11n_obb_full/weights/best.pt` |
| 原图级评估报告 | `outputs/detector/yolo11n_obb_full_test_original/original_eval_report.json` |
| Detector ROI metadata | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl` |
| Detector ROI labels | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.jsonl` |
| GT ROI crops | `data/roi_crops_gt/dota_v1_lite_300_100_100/` |
| 多出口模型权重 | `outputs/multi_exit/formal/best.pt` |
| 旧 profiling 表 | `data/profiling/dota_v1_lite_300_100_100/test_full/roi_profile_smoke.csv` |
| 新 image-feature profiling 表 | `data/profiling/dota_v1_lite_300_100_100/test_full_image_features/roi_profile_smoke.csv` |
| 旧质量代理 | `outputs/proxy/test_full/quality_proxy.joblib` |
| 新轻量质量代理 | `outputs/proxy/lightweight_image_features/quality_proxy.joblib` |
| 新 proxy 对比报告 | `outputs/proxy/lightweight_image_features/summary.json` |
| 新 deterministic sanity | `outputs/simulation/deterministic_lightweight_proxy_all/summary_all.json` |
| Top-k 20k no-shaping formal | `outputs/rl/ppo_topk_lightweight_proxy_no_shaping_20k/summary.json` |
| Top-k 20k shaping formal | `outputs/rl/ppo_topk_lightweight_proxy_formal_20k/summary.json` |
| Legal 20k shaping formal | `outputs/rl/ppo_legal_lightweight_proxy_formal_20k/summary.json` |

## 9. 组会汇报建议

建议按以下顺序讲：

1. 先讲研究目标：不是完整图像传输，而是 ROI 级语义通信和协同推理调度。
2. 再讲实验闭环：DOTA 到 ROI，到多出口 profiling，到 proxy，到 LEO 调度环境，到 PPO formal。
3. 然后讲最新进展：image-feature MLP 显著提升 proxy，top-k 20k 下 PPO 已超过强基线。
4. 接着讲原文对齐：去模型切分、质量代理、SNR-AMC 查表、定频队列、网格 AoSI 都已经落地。
5. 最后讲限制：top-k 仍需明确方法身份，legal PPO 尚不稳，链路和数据隔离还不是论文最终级别。

可以用下面这段作为汇报结尾：

```text
目前项目已经完成可运行闭环，并且新质量代理和 top-k PPO ablation 给出了积极信号。
现阶段最可靠的结论是：图像质量特征能明显提升 proxy，动作空间剪枝能显著改善 PPO 学习。
但最终论文结论还需要严格数据隔离、更大规模 formal、高保真链路，以及对 top-k 是否属于主方法的明确定位。
```
