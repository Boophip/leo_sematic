# LEO 遥感图像语义通信项目阶段性实验报告

报告日期：2026-06-16

## 1. 实验背景与研究目标

本项目面向低轨卫星（LEO）遥感图像语义通信任务，目标是在带宽、算力和链路状态动态变化的条件下，以 ROI 为调度粒度，优先保障关键目标被及时、正确地处理。根据 `PROJECT_CONTEXT.md` 与 `project.pdf`，完整研究闭环包括：

```text
DOTA 大幅遥感图像
    -> 轻量级 ROI 检测
    -> ROI 语义价值计算
    -> 多出口任务模型
    -> 离线 Profiling
    -> MLP 任务质量代理
    -> LEO 链路、队列与 AoSI 仿真
    -> 强化学习联合调度
```

截至当前版本，项目已经完成并验证的重点集中在第一阶段“可信视觉数据”和 ROI 元数据生成：包括 DOTA 原图级数据划分、重叠切片、YOLO-OBB 数据转换、轻量旋转目标检测器训练与评估、切片检测结果恢复到原图坐标、全局旋转 NMS、ROI 语义价值与 AoSI 网格输入生成，以及基于 DOTA GT 的 ROI 离线质量标签。

本阶段修改性质属于“仅改变工程实现”和“实验基线输入生成”，未改变论文公式。

## 2. 原文思路对照

`project.pdf` 的核心工程取向是：使用轻量 ROI 检测器在源卫星侧先提取候选语义区域，避免传输完整大图；删除复杂的模型中间切分，只保留本地早退与跨星完整卸载；用离线 MLP 质量代理约束调度器，防止其为了降低时延和能耗而过度压缩或过早退出；用静态空间网格 AoSI 替代难以落地的目标级跨帧跟踪。

当前代码实现与原文保持一致的部分包括：

- 以 DOTA 原始大图为划分单位，避免切片后随机划分造成数据泄漏。
- 使用 YOLO11n-OBB 作为轻量级旋转 ROI 检测器。
- 将切片检测结果恢复到原图坐标后再做全局旋转框 NMS。
- ROI 语义价值使用类别优先级、检测置信度和面积占比加权截断公式。
- AoSI 网格基于原图坐标系的固定 8x8 网格，而不是检测切片网格。
- ROI 离线标签只作为后续质量代理和 profiling 的监督信息，不把单个 ROI 质量称为 mAP。

当前尚未实现的原文内容包括：多出口精细任务模型、压缩等级 profiling、MLP 质量代理、LEO 链路与 SNR-AMC 查表、计算队列/能耗模型、AoSI 跨时隙更新、PPO 或 MADRL 调度训练。因此本报告中的数值结论只覆盖视觉前端与 ROI 数据准备阶段。

## 3. 已完成代码与数据产物

当前代码结构已经形成清晰的工程边界：

| 位置 | 已有内容 | 作用 |
|---|---|---|
| `src/data/dota_lite_split.py` | DOTA 原图级子集划分 | 保证 train/val/test 按原图 ID 隔离 |
| `src/data/dota_tiling.py` | 大图重叠切片与坐标映射 | 生成 1024 尺寸检测切片，并保留恢复信息 |
| `src/data/dota_yolo_obb.py` | DOTA 切片转 YOLO-OBB 格式 | 生成 Ultralytics 可训练数据集 |
| `src/detector/yolo_obb_reporting.py` | YOLO 训练报告汇总 | 固化模型参数、速度、指标与环境信息 |
| `src/detector/dota_original_eval.py` | 原图级恢复、NMS 与 AP50 评估 | 将切片预测还原到 DOTA 原图评价口径 |
| `src/data/roi_metadata.py` | ROI 元数据、语义价值、裁剪和网格汇总 | 生成调度环境的 ROI 输入 |
| `src/data/roi_labeling.py` | ROI 与 DOTA GT 匹配标注 | 为后续质量代理/profiling 准备监督标签 |
| `scripts/01` 到 `scripts/09` | 对应流水线入口 | 可复现实验步骤 |
| `tests/` | 30 个单元测试 | 验证公式、边界和数据 schema |

## 4. 数据集准备

当前使用 DOTA-v1.0 的可用训练图像构建了一个 500 张原图的实验子集，随机种子为 42，划分如下：

| 划分 | 原图数 | 原始目标数 |
|---|---:|---:|
| train | 300 | 20741 |
| val | 100 | 6922 |
| test | 100 | 7003 |

切片配置为 `tile_size=1024`、`overlap=200`、`min_visible_ratio=0.7`。切片后数据规模如下：

| 划分 | 切片数 | 原始目标数 | 被覆盖目标数 | 切片内目标数 | 空标签切片 |
|---|---:|---:|---:|---:|---:|
| train | 3926 | 20741 | 20724 | 38835 | 1615 |
| val | 1295 | 6922 | 6917 | 12525 | 551 |
| test | 1310 | 7003 | 7003 | 13912 | 580 |

转换为 YOLO-OBB 后，过滤 `difficult=2` 的部分目标，保留用于检测训练的目标数如下：

| 划分 | 图像数 | 保留目标数 | 空标签数 |
|---|---:|---:|---:|
| train | 3926 | 36750 | 1720 |
| val | 1295 | 11837 | 593 |
| test | 1310 | 13182 | 617 |

## 5. 检测器训练与评估

### 5.1 过拟合冒烟实验

项目先完成了 32 张图像的 YOLO11n-OBB 同集过拟合测试，用于确认数据格式、标注方向、类别顺序和训练脚本可用。该实验不是正式指标。验证结果为：

| 指标 | 数值 |
|---|---:|
| Precision | 0.923 |
| Recall | 0.962 |
| mAP50 | 0.979 |
| mAP50-95 | 0.866 |

该结果说明 YOLO-OBB 数据转换和训练链路基本正确。

### 5.2 正式切片级训练

正式模型使用 `yolo11n-obb.pt` 初始化，在 1024 切片上训练 100 epoch。模型参数量约 2.66M，计算量约 16.83 GFLOPs，最佳权重位于：

```text
outputs/detector/yolo11n_obb_full/weights/best.pt
```

验证集聚合结果：

| 指标 | 数值 |
|---|---:|
| Precision | 0.803 |
| Recall | 0.728 |
| mAP50 | 0.775 |
| mAP50-95 | 0.606 |

独立 test 切片级结果：

| 指标 | 数值 |
|---|---:|
| Precision | 0.766 |
| Recall | 0.704 |
| mAP50 | 0.721 |
| mAP50-95 | 0.545 |

test 集单切片速度记录为：预处理 2.88 ms、推理 6.27 ms、后处理 3.12 ms。该速度是 Ultralytics 切片级评估环境下的统计值，可作为后续星载轻量检测开销建模的初步参考。

![YOLO11n-OBB 训练曲线](../outputs/detector/yolo11n_obb_full/results.png)

![YOLO11n-OBB 验证集 PR 曲线](../outputs/detector/yolo11n_obb_full/BoxPR_curve.png)

![YOLO11n-OBB 归一化混淆矩阵](../outputs/detector/yolo11n_obb_full/confusion_matrix_normalized.png)

![YOLO11n-OBB test 可视化预测样例](../outputs/detector/yolo11n_obb_full_test/val_batch0_pred.jpg)

从类别表现看，切片级 test mAP50 较高的类别包括 plane 0.976、ship 0.960、large-vehicle 0.943、small-vehicle 0.917、tennis-court 0.901；较弱类别包括 helicopter 0.036、soccer-ball-field 0.307、roundabout 0.389。低样本类别与小目标/形态复杂类别仍是后续优化重点。

## 6. 原图级恢复、NMS 与 ROI 质量

为了符合论文中“切片只服务检测，AoSI 与 ROI 应回到原图/物理坐标系”的要求，项目实现了从切片预测到原图坐标的恢复，并对同一原图、同一类别执行全局旋转 NMS。

原图级 DOTA-style AP50 本地评估配置：

| 项 | 值 |
|---|---|
| split | test |
| confidence | 0.001 |
| tile_iou | 0.7 |
| nms_iou | 0.1 |
| eval_iou | 0.5 |
| limit_images | null |

原图级聚合结果：

| 指标 | 数值 |
|---|---:|
| Precision | 0.333 |
| Recall | 0.928 |
| mAP50 | 0.691 |
| True Positives | 6352 |
| False Positives | 12700 |

该结果体现了当前检测器前端偏向高召回。对于本项目而言，高召回是合理的阶段性目标，因为 ROI 一旦漏检，后续调度无法补救；但假阳性数量偏多会增加后续 profiling、质量代理和调度环境负载，后续需要通过阈值、NMS、类别校准或质量代理筛选继续控制。

原图级召回表现较好的类别包括 large-vehicle 0.968、plane 0.966、basketball-court 0.962、small-vehicle 0.957、ship 0.947；较弱类别包括 helicopter 0.118、roundabout 0.565、soccer-ball-field 0.625。

## 7. ROI 元数据与 AoSI 网格输入

基于原图级 NMS 结果，当前已经生成 test split 的 ROI 元数据、旋转裁剪图像和 AoSI 网格汇总：

```text
data/roi_metadata/dota_demo_1024_o200/test/
```

生成规模：

| 项 | 数值 |
|---|---:|
| 原图数 | 100 |
| ROI 数 | 19185 |
| 裁剪图数 | 19185 |
| 网格单元数 | 6400 |
| 非空网格单元数 | 3252 |

ROI 元数据 schema 包括 `roi_id`、`image_id`、`source_image_path`、`crop_path`、`predicted_class`、`class_id`、`confidence`、`global_obb`、`center_x/y`、`area_ratio`、`grid_row/col/id`、`semantic_value`、`class_priority`、`difficult`、`source_tile_id`。

当前语义价值公式与项目公式一致：

```text
v_k = clip(w_class * class_priority(y_k)
           + w_confidence * confidence_k
           + w_area * area_ratio_k,
           0, 1)
```

当前权重为：

| 权重 | 数值 |
|---|---:|
| w_class | 0.50 |
| w_confidence | 0.35 |
| w_area | 0.15 |

AoSI 网格使用原图坐标系下的 8x8 固定网格，每个 ROI 按 OBB 中心点唯一归属。网格综合语义价值采用概率 OR 聚合：

```text
V_g = 1 - product(1 - v_k), k belongs to grid g
```

这部分实现为后续调度环境提供了可直接采样的 ROI 业务流和空间语义状态。

## 8. ROI 离线质量标签

项目已经实现了 ROI 与 DOTA 原始 GT 的旋转 IoU 匹配，阈值为 0.5。匹配策略采用 DOTA/VOC 风格的一对一匹配，并区分：

- `true_positive`
- `ignored_difficult`
- `class_mismatch`
- `background_fp`
- `duplicate_detection`

test split 的标签统计如下：

| 项 | 数值 |
|---|---:|
| ROI 总数 | 19185 |
| GT 数 | 6842 |
| ignored GT | 161 |
| true_positive | 6352 |
| false_positive | 12700 |
| ignored_prediction | 133 |
| class_mismatch | 310 |
| background_fp | 12390 |
| matched_any_gt | 6795 |
| class_correct | 6485 |

质量标签分布：

| 标签 | 数量 |
|---|---:|
| true_positive | 6352 |
| ignored_difficult | 133 |
| class_mismatch | 310 |
| background_fp | 12390 |

这部分结果不应被解释为单 ROI mAP，而是后续多出口模型、压缩 profiling 和 MLP 质量代理的离线监督材料。

## 9. 测试与可复现性

本次报告生成前运行了完整单元测试：

```text
python -m unittest discover -s tests -v
```

结果：

```text
Ran 30 tests in 0.580s
OK
```

测试覆盖内容包括：

- DOTA 子集划分的确定性、类别覆盖和数据隔离。
- 切片坐标恢复、边界填充、部分目标裁剪和 `difficult` 标记。
- YOLO-OBB 标签归一化、类别顺序和 smoke 数据选择。
- 原图级坐标恢复、全局旋转 NMS、困难样本忽略、重复预测计数。
- ROI 语义价值公式、面积占比、网格归属、概率 OR 聚合和裁剪输出。
- ROI 标注的 TP、class mismatch、background FP 和 ignored difficult 逻辑。

## 10. 阶段性结论

当前项目已经完成从 DOTA 原图到调度可用 ROI 元数据的前端闭环。最重要的阶段性成果是：数据划分遵守原图隔离；检测器训练链路可用；切片预测可以恢复到原图坐标并进行全局旋转框 NMS；ROI 已经带有语义价值、网格归属、裁剪路径和离线质量标签。

从实验结果看，切片级 YOLO-OBB 在 test 上达到 mAP50 0.721、mAP50-95 0.545；原图级恢复后的 AP50 为 0.691，召回率达到 0.928。该现象符合当前“先保障关键 ROI 不漏检”的阶段目标，但也暴露出假阳性偏多的问题，后续会直接影响 ROI 业务流规模和调度压力。

当前尚不能宣称完成论文级联合调度实验，因为多出口模型、压缩 profiling、质量代理、LEO 链路队列仿真和强化学习策略尚未实现。下一阶段应优先沿着原文路线推进“多出口模型 + 压缩等级 profiling + MLP 质量代理”，使调度环境能够使用动作相关的质量预测，而不是固定质量或手工标签。

## 11. 后续建议

1. 在现有 ROI 裁剪集上构建多出口任务模型，输出每个出口的推理质量、置信度和时延。
2. 对 ROI 裁剪图像执行多级压缩与缩放，生成 `ROI x exit x beta` profiling 表。
3. 训练轻量 MLP 质量代理，并在独立 test split 上报告 MAE、MSE、R-squared 和分组误差。
4. 用当前 ROI 元数据作为输入，建立最小可验证的确定性调度环境，先实现 SNR-AMC 查表、通信时延、计算队列和 AoSI 跨时隙更新。
5. 在环境因果关系通过测试后，再训练 PPO 或后续 MADRL 策略，并与 Random、Local-Deep、Best-SNR、Semantic-Greedy、No-AoSI 等基线比较。

## 附录：主要产物路径

| 产物 | 路径 |
|---|---|
| DOTA 子集摘要 | `data/DOTA-demo/summary.json` |
| 切片摘要 | `data/tiles/dota_demo_1024_o200/summary.json` |
| YOLO-OBB 数据集配置 | `data/yolo_obb/dota_demo_1024_o200/dataset.yaml` |
| 正式检测器权重 | `outputs/detector/yolo11n_obb_full/weights/best.pt` |
| 切片级 test 评估报告 | `outputs/detector/yolo11n_obb_full_test/evaluation_report.json` |
| 原图级评估报告 | `outputs/detector/yolo11n_obb_full_test_original/original_eval_report.json` |
| 原图级 NMS 预测 | `outputs/detector/yolo11n_obb_full_test_original/nms_predictions.jsonl` |
| ROI 元数据 | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl` |
| ROI 标注数据 | `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.jsonl` |
| ROI 网格汇总 | `data/roi_metadata/dota_demo_1024_o200/test/grid_summary.json` |
