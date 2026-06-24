# LEO 遥感图像语义通信项目组会进展总结

报告日期：2026-06-22

本文用于组会汇报，重点回答四个问题：

1. 目前已经做了哪些工作。
2. 整个实验流程现在是什么样的。
3. 还有哪些部分没有完成或结论不稳。
4. 当前实现与 `project.pdf` 原文思路是否一致。

文中对常见概念和技术指标做了简要注释，方便汇报时解释背景。

## 1. 一句话结论

项目已经从单个视觉模块推进到“视觉前端 + 质量建模 + 卫星调度环境 + PPO 训练评估”的可运行闭环。当前代码能够复现完整流程，单元测试通过；但最终论文级结论仍未形成，主要瓶颈在质量代理精度、PPO 主设置表现和高保真链路仿真。

更具体地说：

- 已完成 DOTA 原图级划分、重叠切片、YOLO11n-OBB 检测、原图级坐标恢复和旋转框 NMS。
- 已完成 ROI 元数据、语义价值、AoSI 网格归属、GT ROI crops、多出口模型、压缩 profiling 和 MLP 质量代理。
- 已完成确定性卫星仿真，包括 SNR-AMC、不可见链路非法动作、固定频率队列、能耗、AoSI 和多种基线策略。
- 已完成单智能体 PPO smoke、pilot 和 formal runner，并完成 `legal` 与 `feasible-topk` 两类候选动作实验。
- 当前 `legal` 主设置下 Proposed-RL 尚未稳定超过强启发式基线；`feasible-topk` 表现更好，但应作为候选动作剪枝消融，而不是直接作为主方法结论。

## 2. 总体流程

当前实验闭环如下：

```text
DOTA 原图
  -> 原图级 train/val/test 划分
  -> 重叠切片与 YOLO-OBB 数据转换
  -> YOLO11n-OBB 训练与测试
  -> 切片检测结果恢复到原图坐标
  -> 全局旋转框 NMS 去重
  -> 生成 detector ROI metadata
  -> 计算 ROI 语义价值与 AoSI 网格编号
  -> 将 detector ROI 与 DOTA GT 匹配，得到 TP/FP 标签
  -> 构建 GT ROI crops
  -> 训练多出口任务模型
  -> 遍历 ROI x exit x compression，生成离线 profiling 表
  -> 用 profiling 表训练 MLP 质量代理
  -> 构建 LEO 链路、计算队列、能耗、AoSI 调度环境
  -> 运行确定性基线
  -> 训练和评估 PPO 调度策略
```

这个流程可以分成两个大的阶段：

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| 离线视觉建模 | 提供 ROI、任务质量和动作代价的可查表数据 | 已打通 |
| 在线调度仿真 | 在链路和算力动态变化下选择处理动作 | 已打通，但 PPO 结论不稳 |

### 2.1 为什么先做离线视觉建模

原文强调在线调度不能在每个 step 运行完整视觉模型，因为星载计算资源有限，强化学习训练也需要大量环境交互。因此当前工程做法是：

- 在线环境只读 ROI metadata、profiling 表和质量代理。
- 多出口模型和图像压缩只在离线 profiling 阶段运行。
- PPO 训练时不再调用 YOLO 或多出口 CNN，只调用轻量表格代理和仿真模型。

这与原文“在轨极速映射”和“不在每步做昂贵非线性求解”的思想一致。

### 2.2 为什么先做单智能体 PPO

`project.pdf` 最终倾向单层 MADRL，但当前先实现单智能体 PPO 是合理的过渡步骤。原因是：

- 单智能体环境更容易验证动作是否真实影响质量、时延、队列、能耗和 AoSI。
- 当前关键瓶颈仍在 proxy 与 action space，而不是多智能体协作本身。
- 若单智能体闭环尚未稳定，直接升级 MADRL 会放大调试难度。

## 3. 原文思路与当前实现的关系

`project.pdf` 的核心问题是：低轨卫星遥感图像很大，但真正有业务价值的目标很稀疏。系统不应追求完整图像无损回传，而应优先保证关键 ROI 被及时、正确处理。

当前实现与原文的主要对应关系如下。

| 原文设计 | 当前实现 | 一致性判断 |
|---|---|---|
| 源卫星侧轻量 ROI 解析器 | YOLO11n-OBB 检测器 | 一致 |
| ROI 语义价值连续建模 | 类别优先级、置信度、面积比例加权 | 一致 |
| 去掉模型中间切分 | 只保留本地早退和跨星完整卸载 | 一致 |
| 多出口 early-exit | Tiny multi-exit CNN | 一致，当前为简化模型 |
| 离线 profiling | 遍历 ROI、exit、compression | 一致 |
| MLP 质量代理防止“作弊” | `outputs/proxy/test_full/quality_proxy.joblib` | 已实现但精度需提升 |
| 固定功率 + SNR-AMC 查表 | `src/simulation/link.py` | 一致 |
| 定频计算队列 | `src/simulation/compute.py` | 一致 |
| 静态网格 AoSI | `src/simulation/aosi.py` 与 ROI grid | 一致 |
| Lyapunov 虚拟队列 | 环境中已有质量和时延虚拟队列 | 部分实现 |
| 单层 MADRL | 当前为单智能体 PPO | 未完成 |
| STK/ns-3 高保真在轨仿真 | 当前为 synthetic trace | 未完成 |

### 3.1 与原文不同但需要说明的工程设定

当前实现中有几处工程化设定，原文没有给出具体数值，但不改变研究公式。

| 工程设定 | 当前取值或做法 | 说明 |
|---|---|---|
| 压缩等级 | `local`, `beta_0..3` | 原文保留压缩动作维度，但未规定 JPEG 参数 |
| 图像数据规模 | DOTA-v1.0 lite 子集，500 张原图 | 用于先打通流程和控制实验成本 |
| 链路轨迹 | synthetic visibility/SNR trace | 用于可复现验证，尚非高保真在轨链路 |
| PPO candidate mode | `legal`, `feasible-topk` | 用于处理固定 action space 与动态合法动作集合的矛盾 |
| 多出口模型 | Tiny multi-exit CNN | 当前先验证早退与质量差异，不是最终大模型 |

这些设定在组会上需要主动说明，避免被理解为已经完成高保真在轨实验或最终算法定型。

## 4. 阶段一：DOTA 数据划分与切片

### 4.1 已完成工作

已基于 DOTA-v1.0 可用训练图像构建 500 张原图子集，随机种子为 42。划分在原图级完成，保证同一原图产生的切片和 ROI 不会跨 split 泄漏。

| Split | 原图数 | 原始目标数 |
|---|---:|---:|
| train | 300 | 20741 |
| val | 100 | 6922 |
| test | 100 | 7003 |

切片配置如下：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `tile_size` | 1024 | 每个检测切片的边长 |
| `overlap` | 200 | 相邻切片重叠像素数 |
| `min_visible_ratio` | 0.7 | 目标被切片截断后，至少保留多少比例才写入该切片 |

切片结果如下：

| Split | 切片数 | 切片内目标数 | 空标签切片 | dropped source objects |
|---|---:|---:|---:|---:|
| train | 3926 | 38835 | 1615 | 17 |
| val | 1295 | 12525 | 551 | 5 |
| test | 1310 | 13912 | 580 | 0 |

### 4.2 为什么要做原图级划分

DOTA 是大幅遥感图像。若先切片再随机划分，同一原图的相邻切片可能分别进入训练集和测试集，模型会在测试时看到与训练图高度相似的区域，导致指标虚高。当前按原图 ID 先划分，再切片，符合 `PROJECT_CONTEXT.md` 中的数据泄漏约束。

### 4.3 概念和指标注释

| 名称 | 注释 |
|---|---|
| DOTA | 遥感目标检测数据集，目标多为旋转框标注，类别包括 ship、plane、vehicle、harbor 等 |
| 原图级划分 | 以原始大图为单位划分 train/val/test，而不是以切片或 ROI 为单位 |
| 重叠切片 | 大图切成小图时保留重叠区域，降低目标刚好落在边界被切断的概率 |
| 空标签切片 | 切片中没有可用目标。保留一部分空切片有助于检测器学习背景 |
| dropped source objects | 因切片可见比例不足等原因没有被任何切片有效覆盖的原始目标 |

## 5. 阶段二：YOLO11n-OBB ROI 检测前端

### 5.1 已完成工作

已完成 YOLO11n-OBB formal 训练、独立切片级 test 评估和原图级恢复/NMS 评估。

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

### 5.2 如何理解这组指标

切片级指标看的是每个 1024 x 1024 tile 上的检测效果；原图级指标看的是把所有 tile 的检测结果恢复到原图坐标、再做全局去重后的效果。原图级更接近后续调度系统实际看到的 ROI 流。

目前最关键的结论是：

- 原图级 recall 很高，说明关键目标大多能进入后续调度流程。
- 原图级 precision 偏低，说明误检较多，会增加无效 ROI 和调度负担。
- 这符合当前阶段目标：先保证“不漏关键目标”，再优化 FP 和系统负载。

### 5.3 概念和指标注释

| 名称 | 注释 |
|---|---|
| YOLO11n-OBB | Ultralytics 的轻量级旋转框检测模型，`n` 表示 nano 规模，OBB 表示 oriented bounding box |
| OBB | 旋转目标框。遥感图像中目标方向任意，旋转框比水平框更贴合船、飞机、车辆等目标 |
| Precision | 检测出的目标中有多少是真的。公式近似为 `TP / (TP + FP)` |
| Recall | 真实目标中有多少被检测到。公式近似为 `TP / (TP + FN)` |
| mAP50 | IoU 阈值为 0.5 时，各类别 AP 的平均值 |
| mAP50-95 | IoU 从 0.50 到 0.95 多个阈值下 mAP 的平均值，比 mAP50 更严格 |
| TP | true positive，检测框与 GT 匹配且类别正确 |
| FP | false positive，检测器报出的目标没有匹配到正确 GT |
| FLOPs | 一次前向推理大约需要的浮点运算量，用于衡量计算开销 |
| NMS | non-maximum suppression，非极大值抑制，用于去掉多个重叠检测框中的重复结果 |

### 5.4 与原文的关系

原文假设前端轻量检测器召回率较高，因为 ROI 一旦漏检，后续调度无法补救。当前原图级 `Recall=0.9284` 基本支撑这一假设。但 precision 偏低说明还需要校准，否则调度环境会处理大量背景 FP。

## 6. 阶段三：ROI 元数据、语义价值与 AoSI 网格

### 6.1 已完成工作

已根据原图级 NMS 结果生成 detector ROI metadata。每条 ROI 记录包含：

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
| true positive | 6352 |
| false positive | 12700 |
| class mismatch | 310 |
| background FP | 12390 |
| ignored prediction | 133 |

### 6.2 ROI 语义价值

当前 ROI 语义价值公式为：

```text
v_k = clip(
    w_class * class_priority(y_k)
    + w_confidence * confidence_k
    + w_area * area_ratio_k,
    0,
    1
)
```

当前工程权重：

```text
w_class = 0.5
w_confidence = 0.35
w_area = 0.15
```

解释：

- `class_priority(y_k)` 表示类别优先级。例如 ship、plane、harbor 等更重要类别优先级更高。
- `confidence_k` 表示检测器对该 ROI 类别的置信度。
- `area_ratio_k` 表示 ROI 面积占原图面积的比例。
- `clip` 将结果截断到 `[0,1]`，避免超出语义价值定义域。

语义价值只表示“这个 ROI 重要不重要”，不表示“处理是否正确”。最终是否做对，需要质量代理预测 `q_k`。

### 6.3 AoSI 网格归属

当前将 ROI 根据中心点归属到固定网格。DOTA 切片不等于 AoSI 网格，两者用途不同：

| 类型 | 用途 | 是否固定空间语义 |
|---|---|---|
| 检测切片 | 为了让大图适配检测器输入尺寸 | 否 |
| AoSI 网格 | 表示固定物理空间的新鲜度和语义变化 | 是 |

网格综合语义价值使用概率 OR：

```text
V_g = 1 - product(1 - v_k), k belongs to grid g
```

该公式的好处是：只要网格内有一个高价值 ROI，网格价值就会较高；多个中等价值 ROI 也能累积，但总值不会超过 1。

### 6.4 概念和指标注释

| 名称 | 注释 |
|---|---|
| ROI | region of interest，感兴趣区域。这里指从遥感图像中检测出的候选目标区域 |
| semantic value | ROI 固有语义价值，反映业务重要性，不反映最终识别正确性 |
| class priority | 类别优先级。由业务需求设定，例如船、港口、飞机等可设为高价值 |
| area ratio | ROI 面积除以原图面积。面积较大的目标可能承载更多业务价值 |
| grid_id | ROI 所属 AoSI 网格编号，用于后续空间新鲜度更新 |
| background FP | 检测器误报，没有匹配到任何真实目标 |
| class mismatch | 检测框匹配到真实目标，但预测类别不对 |

## 7. 阶段四：GT ROI crops 与多出口任务模型

### 7.1 已完成工作

当前使用 GT ROI crops 训练多出口任务模型。这样做是为了先获得干净的监督信号，避免 detector FP 直接污染任务模型训练。

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

### 7.2 如何理解多出口结果

多出口模型有多个提前输出分支。浅层出口计算快，但特征表达不足；深层出口计算慢，但准确率高。当前结果符合这一趋势：

- Exit 1 准确率较低，适合非常简单或低价值 ROI。
- Exit 2 到 Exit 4 准确率逐渐提升。
- Exit 4 最高，代表最深层推理，但计算成本也最高。

这为调度器提供了真实 trade-off：不同 ROI 不一定都要跑到最深出口。

### 7.3 概念和指标注释

| 名称 | 注释 |
|---|---|
| GT ROI crops | 从 DOTA ground truth 标注中裁出的真实目标图像块，用于监督训练 |
| Early Exit | 早退机制。模型中间层也能输出结果，用较低计算成本处理简单样本 |
| Exit Accuracy | 某个出口独立预测类别的准确率 |
| mean accuracy | 多个出口准确率的平均值，用于粗略衡量整体多出口模型质量 |
| logits | 模型最后一层输出的未归一化分数，训练交叉熵时应输入 logits，而不是 softmax 概率 |

### 7.4 与原文的关系

原文强调 early-exit 可以让系统根据计算预算动态选择推理深度。当前多出口模型已经提供了这种动作维度。当前模型是轻量 baseline，不代表最终视觉模型上限，但足以支撑调度闭环验证。

## 8. 阶段五：压缩策略与离线 Profiling

### 8.1 已完成工作

当前压缩策略如下：

| Level | 编码方式 | 缩放 | 用途 |
|---|---|---:|---|
| `local` | none | 1.00x | 本地处理，不产生通信 payload |
| `beta_0` | PNG lossless | 1.00x | 无损卸载质量上界 |
| `beta_1` | JPEG quality 90 | 1.00x | 高质量卸载 |
| `beta_2` | JPEG quality 70 | 0.75x | 中等带宽压力 |
| `beta_3` | JPEG quality 45 | 0.50x | 强压缩低带宽场景 |

已使用 test GT ROI crops、formal multi-exit checkpoint 和 5 种压缩/本地等级生成 profiling 表。

| 项 | 数值 |
|---|---:|
| ROI 数 | 6842 |
| Exit 数 | 4 |
| 压缩/本地等级 | 5 |
| profiling 行数 | 136840 |

行数计算为：

```text
6842 ROI x 4 exits x 5 compression/local levels = 136840 rows
```

### 8.2 Profiling 表记录什么

profiling 表的核心字段包括：

```text
roi_id
image_id
predicted_class
detector_confidence
area_ratio
semantic_value
grid_id
class_priority
quality_label
exit_level
compression_level
compressed_bytes
encode_ms
decode_ms
inference_ms
task_quality
```

这些字段分别支持三类在线查询：

- 质量查询：某个 ROI 在某个出口和压缩等级下预期能做多好。
- 通信查询：压缩后需要传多少字节，传输时延大概是多少。
- 计算查询：执行到某个出口大概需要多少推理时间和 cycles。

### 8.3 概念和指标注释

| 名称 | 注释 |
|---|---|
| Profiling | 离线枚举动作组合并记录质量、时延、数据大小的过程 |
| compression level | 压缩等级。压缩越强，通信量越小，但可能降低任务质量 |
| compressed bytes | ROI 压缩后的字节数，用于计算通信时延 |
| encode/decode ms | 压缩编码和解码耗时 |
| inference ms | 多出口模型推理到指定出口的耗时 |
| task quality | 当前工程中用于训练 proxy 的任务质量标签，不能直接等同于数据集级 mAP |

### 8.4 与原文的关系

原文只规定压缩等级是动作维度之一，没有规定具体 JPEG quality 或缩放比例。当前 `beta_0..3` 是首版可复现实验设定，不改变原文公式。

同时，原文已经取消模型中间切分。因此当前实现中：

- 本地处理不压缩，不产生通信 payload。
- 跨星卸载压缩原始 ROI crop，目标卫星解码后从模型入口运行到指定 exit。

## 9. 阶段六：MLP 质量代理

### 9.1 已完成工作

已训练 MLP 质量代理，模型路径为：

```text
outputs/proxy/test_full/quality_proxy.joblib
```

当前输入字段排除了明显泄漏字段：

```text
quality_label
task_quality
predicted_class_id
exit_confidence
```

这些字段不能作为 proxy 输入，因为它们直接或间接包含了模型真实输出结果。如果使用它们，在线调度会看到实际运行时不可能提前知道的信息。

当前训练/测试划分如下：

| Split | 行数 | 图像数 |
|---|---:|---:|
| train | 117460 | 75 |
| test | 19380 | 25 |

指标如下：

| Split | MAE | MSE | R2 |
|---|---:|---:|---:|
| train | 0.1319 | 0.0380 | 0.6692 |
| test | 0.1893 | 0.0687 | 0.4607 |

### 9.2 为什么质量代理很关键

如果调度奖励只看 ROI 语义价值 `v_k`，智能体可能学到“作弊”策略：

- 对高价值 ROI 使用极强压缩，减少通信开销。
- 总是选择最浅出口，减少计算时延。
- 这样时延和能耗低，但任务根本没有做对。

质量代理 `q_k` 的作用是估计“该动作组合下任务是否还能保持有效质量”。调度奖励使用 `v_k x q_k`，系统才会同时关心“目标重要”和“处理正确”。

### 9.3 指标注释

| 名称 | 注释 | 当前解读 |
|---|---|---|
| MAE | mean absolute error，平均绝对误差，越小越好 | test 为 0.1893，说明平均预测偏差仍较明显 |
| MSE | mean squared error，平方误差均值，对大误差更敏感，越小越好 | test 为 0.0687 |
| R2 | 决定系数，表示模型解释目标方差的能力，越接近 1 越好 | test 为 0.4607，仍偏弱 |
| grouped error | 按类别、exit、compression 分组统计误差 | 用于发现某些类别或动作组合误差特别大 |

### 9.4 当前问题

质量代理已经打通，但还不够强。主要问题是：

- test `R2=0.4607`，解释力不足。
- 当前 profiling 主要来自 `test_full` 内部分组切分，最终论文实验还需要更严格的 train/val/test profiling 隔离。
- 不同类别和出口下误差可能不均衡，会影响调度策略对高价值 ROI 的判断。

这是下一阶段最优先的问题之一。

## 10. 阶段七：确定性卫星仿真环境

### 10.1 已完成工作

已实现确定性仿真闭环，主要模块包括：

| 模块 | 作用 |
|---|---|
| `src/simulation/link.py` | SNR-AMC 查表、吞吐率、通信时延、通信能耗 |
| `src/simulation/compute.py` | 固定频率队列、等待时延、计算能耗 |
| `src/simulation/aosi.py` | 网格语义价值、变化因子、AoSI 更新 |
| `src/simulation/episode.py` | 策略动作评估、episode 运行、基线策略 |
| `src/envs/leo_scheduling_env.py` | Gymnasium 环境，供 PPO 训练使用 |

已实现的基线策略包括：

| 策略 | 含义 |
|---|---|
| Random | 随机选择合法动作 |
| Local-Deep | 全部本地处理，并使用最深出口 |
| Local-Adaptive | 全部本地处理，但按质量阈值选择出口 |
| Best-SNR | 卸载到当前 SNR 最好的可见卫星 |
| Semantic-Greedy | 根据语义价值贪心选择动作 |
| No-AoSI | 不考虑 AoSI 的短视策略 |
| AoSI-Greedy | 显式考虑 AoSI 的贪心策略 |

### 10.2 当前确定性基线结果

四个 stress scenario 下当前最好基线如下：

| Scenario | 当前最好基线 | QoE |
|---|---|---:|
| default | No-AoSI | -12.4728 |
| low_snr | No-AoSI | -17.1291 |
| compute_congested | Semantic-Greedy | -35.9335 |
| tight_deadline | No-AoSI | -26.8795 |

这些结果说明不同场景下优势策略不同。例如 compute-congested 场景更考验计算队列，tight-deadline 场景更考验时延约束。

### 10.3 关键公式和含义

通信吞吐率：

```text
throughput = bandwidth * spectral_efficiency(SNR)
```

通信时延：

```text
transmission_delay = compressed_size / throughput
```

计算队列：

```text
Q_i(t + 1) = max(Q_i(t) - f_i * delta_t, 0) + arrival_cycles_i(t)
```

AoSI 语义变化因子：

```text
C_g = alpha + (1 - alpha) * abs(V_g(t) - V_g(t - 1))
```

### 10.4 概念和指标注释

| 名称 | 注释 |
|---|---|
| SNR | signal-to-noise ratio，信噪比。越高通常链路越好 |
| AMC | adaptive modulation and coding，自适应调制编码。当前用 SNR 查表得到频谱效率 |
| spectral efficiency | 频谱效率，单位 bps/Hz。越高表示同样带宽能传更多数据 |
| visibility | 卫星之间当前是否可见。不可见链路不能卸载 |
| compute queue | 计算队列，表示节点上积压的 CPU cycles |
| waiting delay | 等待时延，近似为 `queue_cycles / frequency` |
| energy | 当前包括计算能耗和通信能耗 |
| AoSI | age of semantic information，语义信息年龄。衡量某个空间网格的语义状态是否陈旧 |
| soft reset | 网格被成功处理后，年龄按质量和时延达标程度平滑下降，而不是粗暴清零 |

### 10.5 与原文的关系

这一部分与原文重构方向高度一致。原文明确放弃每 step 香农反算功率优化，改为固定发射功率和 SNR 查表；计算侧也采用固定频率队列；AoSI 从目标级追踪改为静态网格。当前实现正是围绕这三点展开。

## 11. 阶段八：PPO 调度实验

### 11.1 已完成工作

已完成 PPO smoke、pilot 和 formal experiment manager。formal 设置如下：

| 项 | 当前值 |
|---|---:|
| seeds | 42, 43, 44 |
| scenarios | default, low_snr, compute_congested, tight_deadline |
| ROI limit | 512 |
| timesteps per scenario | 5000 |
| eval frequency | 500 |
| checkpoint frequency | 1000 |

PPO 结果使用 canonical slot QoE 评估，而不是直接使用训练 reward。这一点很重要，因为 reward shaping 只用于训练，最终对比应使用统一 QoE。

### 11.2 `legal` candidate mode 结果

`legal` 是当前最接近原文主方法的候选动作设置。环境每步根据当前 ROI、可见链路和本地/卸载约束构造物理合法动作集合，PPO 的 raw action 被映射到这个合法集合。

聚合结果：

| Policy | Mean QoE | Mean Rank | Best Count | Mean Success | Mean Delay ms | Executed Illegal |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-Greedy | -7.8023 | 2.9167 | 0 | 0.9360 | 98.82 | 0 |
| No-AoSI | -10.4192 | 2.4167 | 3 | 0.9214 | 112.85 | 0 |
| Proposed-RL | -14.3282 | 4.3333 | 0 | 0.8449 | 83.49 | 0 |
| AoSI-Greedy | -16.0707 | 2.7500 | 6 | 0.8965 | 150.86 | 0 |

按场景看，Proposed-RL 与当前最好策略仍有差距：

| Scenario | Best Policy | Best Mean QoE | PPO Mean QoE | PPO Mean Gap |
|---|---|---:|---:|---:|
| default | AoSI-Greedy | -2.9404 | -7.0047 | -4.0643 |
| low_snr | AoSI-Greedy | -2.2056 | -8.7541 | -6.5485 |
| compute_congested | Local-Adaptive | -13.5811 | -29.1026 | -15.5214 |
| tight_deadline | No-AoSI | -1.6467 | -12.4516 | -10.8049 |

结论：`legal` 设置下没有物理非法卸载，说明约束机制有效；但 PPO 尚未学到稳定优于强基线的策略。

### 11.3 `feasible-topk` candidate mode 结果

`feasible-topk` 每步只保留 `drop + top-12` 个合法非丢弃候选动作。该设置显著缩小动作空间。

聚合结果：

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

解释：top-k 结果说明 PPO 的学习困难很可能与动作空间过宽、动作索引表达不稳定有关。但 top-k 使用了启发式候选生成，因此应作为消融或改进方向，而不能直接替代主论文一致的 `legal` 设置。

### 11.4 PPO 相关概念和指标注释

| 名称 | 注释 |
|---|---|
| PPO | proximal policy optimization，一种常用强化学习算法，适合离散或连续动作控制 |
| policy | 策略，给定状态后输出动作选择 |
| reward | 训练时给 PPO 的反馈信号，可加入 shaping 项 |
| canonical QoE | 统一评估指标，用于最终策略对比，避免不同训练 reward 导致不可比 |
| candidate mode | 将固定 raw action space 与动态合法动作集合连接起来的机制 |
| legal candidate | 当前所有物理合法动作，包括本地、可见卫星卸载和丢弃 |
| feasible-topk | 从合法动作中按启发式筛选 top-k，降低 PPO 学习难度 |
| executed illegal | 实际执行的非法动作数量，应为 0 |
| Mean Rank | 在多个 seed 和 scenario 中的平均排名，越低越好 |
| Best Count | 成为某个 seed/scenario 最优策略的次数 |
| Mean Success | 满足质量和时延约束的任务比例 |
| Mean Delay ms | 平均端到端任务时延，单位毫秒 |

## 12. 当前验证状态

本次复核使用项目 conda 环境运行：

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

结果：

```text
Ran 87 tests in 3.913s
OK
```

测试覆盖范围包括：

- DOTA 原图级划分。
- DOTA 切片和坐标恢复。
- YOLO-OBB 标签转换。
- 原图级评估、旋转 IoU 和全局 NMS。
- ROI metadata、语义价值、AoSI 网格分配。
- ROI 与 GT 匹配标注。
- GT ROI crops。
- 多出口模型和交叉熵输入检查。
- 压缩与 profiling。
- 质量代理特征防泄漏和 group split。
- SNR-AMC、不可见链路非法动作。
- 固定频率队列、计算能耗和通信能耗。
- AoSI soft reset。
- Gymnasium 环境和 PPO candidate mode。
- PPO formal 聚合报告。

## 13. 当前主要问题

### 13.1 质量代理可信度不足

当前质量代理 test `R2=0.4607`，说明它只能解释一部分任务质量变化。由于调度器依赖 proxy 判断动作质量，这会直接影响 PPO 和启发式策略。

建议下一步：

- 生成 train/val/test 分离的 profiling 表。
- 训练更强的回归器，例如更大 MLP、RandomForest 或 GradientBoosting。
- 按类别、exit 和 compression 分组检查误差。
- 特别关注 high-value ROI 的预测误差。

### 13.2 PPO 主设置表现不稳

`legal` candidate mode 下 Proposed-RL 仍落后强基线，说明当前 PPO 还没有充分利用状态信息。

可能原因：

- 动作空间过宽，raw action 到合法动作的映射学习困难。
- reward shaping 权重不够稳定。
- observation 中某些关键信息表达不足。
- proxy 误差使高质量动作被低估或低质量动作被高估。
- 训练步数和网络超参数仍偏 smoke/pilot 规模。

### 13.3 检测前端 FP 较多

原图级 `Recall=0.9284` 是好事，但 `Precision=0.3334` 表示 FP 很多。对调度系统来说，FP 会占用通信、计算和队列资源。

建议下一步：

- 调整 detector confidence threshold。
- 调整全局旋转 NMS 阈值。
- 对高 FP 类别做类别校准。
- 对比不同阈值下 ROI 数量、召回、proxy 误差和调度 QoE。

### 13.4 链路仍为 synthetic trace

当前链路足以验证因果闭环，但不是高保真在轨仿真。原文中的 STK/ns-3 或真实轨道链路尚未接入。

建议在 PPO 主流程稳定后再接入高保真链路。否则复杂度上升后，很难判断问题来自算法、proxy 还是链路建模。

### 13.5 MADRL 未开始

当前为单智能体 PPO。原文最终目标偏向单层 MADRL，但在单智能体 PPO 未稳定前，不建议马上升级。

## 14. 下一步优先级

### P0：固化当前可复现基线

当前代码和输出已经能支持阶段性汇报。建议先保存现有配置、结果和报告，避免后续调参覆盖。

### P1：提升质量代理

这是当前最重要任务。建议先完成严格 split 的 profiling，再比较多个 proxy baseline。

训练预估：

- 单次 sklearn proxy 训练通常约 1 到 5 分钟。
- 若重新生成大规模 profiling 表，取决于 GPU 和 ROI 数，预计 10 到 60 分钟。

### P2：优化 `legal` PPO 主设置

继续把 `legal` 作为主线。可以尝试：

- 调整 observation，加入更直接的候选动作特征。
- 调整 reward shaping 权重。
- 增加训练步数。
- 改进 action remap 方式。
- 用 top-k 结果指导候选生成，但在报告中标为 ablation。

### P3：降低 detector FP

在不明显损失 high-value recall 的前提下，降低无效 ROI 数量，减轻调度环境压力。

### P4：补正式论文图表

需要形成统一图表体系：

- QoE bar plot。
- QoE CDF。
- success rate。
- delay mean/P95。
- energy。
- traffic bytes。
- AoSI cost。
- training curve。
- scenario heatmap。
- candidate mode 消融图。

### P5：接入高保真链路

在主流程稳定后再接入 STK/TLE/ns-3 或真实链路表，用于更接近论文最终实验。

## 15. 主要产物路径

| 产物 | 路径 |
|---|---|
| 项目背景 | `PROJECT_CONTEXT.md` |
| 原文 PDF | `project.pdf` |
| DOTA 子集摘要 | `data/DOTA-demo/summary.json` |
| 切片摘要 | `data/tiles/dota_demo_1024_o200/summary.json` |
| YOLO-OBB 数据集配置 | `data/yolo_obb/dota_demo_1024_o200/dataset.yaml` |
| 检测器权重 | `outputs/detector/yolo11n_obb_full/weights/best.pt` |
| 切片级 test 报告 | `outputs/detector/yolo11n_obb_full_test/evaluation_report.json` |
| 原图级评估报告 | `outputs/detector/yolo11n_obb_full_test_original/original_eval_report.json` |
| Detector ROI metadata | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl` |
| Detector ROI labels | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.jsonl` |
| GT ROI crops | `data/roi_crops_gt/dota_v1_lite_300_100_100/` |
| 多出口模型权重 | `outputs/multi_exit/formal/best.pt` |
| Profiling 表 | `data/profiling/dota_v1_lite_300_100_100/test_full/roi_profile_smoke.csv` |
| 质量代理 | `outputs/proxy/test_full/quality_proxy.joblib` |
| 确定性仿真报告 | `outputs/simulation/deterministic/comparison_report.md` |
| PPO legal formal | `outputs/rl/ppo_legal_formal/comparison_report.md` |
| PPO top-k formal | `outputs/rl/ppo_topk_formal/comparison_report.md` |

## 16. 组会汇报建议

建议汇报时按以下逻辑讲：

1. 先讲主线：我们不是做完整图像传输，而是做 ROI 级语义通信和协同推理调度。
2. 再讲已完成闭环：DOTA 到 ROI，到多出口 profiling，到 proxy，到 LEO 调度环境，到 PPO formal。
3. 然后讲原文对齐：去模型切分、SNR 查表、定频队列、网格 AoSI、质量代理防作弊都已经落到代码中。
4. 最后讲诚实结论：流程已经跑通，但最终算法优势还不稳。`legal` PPO 仍弱于强基线，top-k 是有效消融但不是主方法。

可以用下面三句话作为汇报总结：

```text
第一，目前项目已经完成从 DOTA ROI 提取到卫星调度 PPO 的可运行闭环，87 个单元测试全部通过。
第二，当前实现与原文重构思路基本一致，核心包括质量代理、去模型切分、SNR-AMC 查表、定频队列和网格 AoSI。
第三，下一步重点不是继续堆复杂度，而是提升质量代理、稳定 legal PPO 主设置，并降低检测 FP 带来的调度负载。
```
