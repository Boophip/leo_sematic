# 面向 LEO 遥感图像语义通信的协同推理与资源调度实验

## 1. 项目简介

本项目研究低轨卫星（Low Earth Orbit, LEO）遥感场景中的任务导向语义通信与协同推理问题。
系统首先从大幅遥感图像中提取具有业务价值的感兴趣区域（Region of Interest, ROI），随后根据
ROI 的语义价值、预期任务质量、星间链路状态、节点计算队列和空间语义新鲜度，联合决定：

- 是否处理或丢弃该 ROI；
- 在源卫星本地处理，还是卸载至协作卫星；
- 使用哪个早退出口（Early Exit）；
- 跨星卸载时使用何种压缩等级。

项目的核心目标不是无损传输完整遥感图像，而是在有限带宽、有限星载算力和动态星间链路下，
优先保证关键目标被及时、正确地处理，并降低通信量、处理时延、能耗和语义信息年龄。

本项目以 DOTA-v1.0 为主要视觉数据集，以强化学习环境模拟 LEO 动态链路与资源调度过程。

实验的四条主线：

1. DOTA 数据预处理与 ROI 检测
2. 多出口模型与质量代理训练
3. 卫星调度环境构建
4. 强化学习训练、基线对比和结果绘图

---

## 2. 研究目标

### 2.1 总体目标

构建并验证一套从遥感目标提取、质量建模到卫星在线调度的完整实验闭环：

```text
DOTA 大幅遥感图像
    -> 轻量级 ROI 检测
    -> ROI 语义价值计算
    -> 多出口精细任务模型
    -> 离线质量 Profiling
    -> MLP 任务质量代理
    -> LEO 链路、队列与 AoSI 仿真
    -> 强化学习联合调度
    -> 与基线方法进行对比评估
```

### 2.2 具体研究问题

1. 轻量级检测器能否以较低星载开销保持足够高的关键目标召回率？
2. 不同 ROI、压缩等级和早退出口组合会如何影响任务质量、数据大小与计算时延？
3. 轻量 MLP 能否准确预测动作组合对应的任务质量？
4. 调度策略能否根据链路、队列和语义价值动态选择本地处理、卸载或丢弃？
5. 引入网格化语义信息年龄（AoSI）后，系统能否更及时地更新高价值或突变区域？
6. 联合调度方法能否优于静态优先级、全部本地、最佳 SNR 卸载等基线？

---

## 3. 系统总体架构

项目分为离线视觉建模和在线调度仿真两个阶段。

### 3.1 离线视觉建模

1. 对 DOTA 原图进行原图级训练、验证和测试划分。
2. 对大幅图像进行重叠切片，并保存切片到原图的坐标映射。
3. 使用轻量 OBB 检测器提取旋转 ROI。
4. 将检测结果恢复至原图坐标，执行旋转框去重，并分配至固定 AoSI 网格。
5. 使用多出口任务模型对 ROI 进行精细识别。
6. 穷举早退出口和压缩等级，生成离线 Profiling 数据。
7. 使用 Profiling 数据训练任务质量代理函数。

### 3.2 在线调度仿真

1. 每个时隙产生一批带有类别、置信度、面积和网格位置的 ROI。
2. 环境更新协作卫星可见性、SNR、计算队列和 AoSI。
3. 调度策略为每个 ROI 选择丢弃、本地处理或跨星卸载动作。
4. 环境计算任务质量、通信时延、计算时延、能耗和约束违背情况。
5. 更新实体计算队列、质量/时延虚拟队列以及网格 AoSI。
6. 使用长期奖励训练和评估强化学习策略。

---

## 4. 技术栈

### 4.1 主要语言与环境

- Python 3.10+
- PyTorch
- CUDA（用于视觉模型训练和 Profiling）
- Windows / Linux 均可运行，当前开发环境以 Windows 为主

### 4.2 视觉与数据处理

- DOTA-v1.0：主要遥感目标检测数据集
- Ultralytics YOLO11n-OBB：轻量级旋转 ROI 检测器
- OpenCV：图像读取、旋转框处理、透视矫正和 ROI 裁剪
- NumPy / Pandas：数值计算和实验数据管理
- scikit-learn：代理模型特征转换与指标计算

### 4.3 多出口与质量代理

- PyTorch / torchvision
- 多出口 ResNet 或其他可早退任务网络
- MLP Task Quality Proxy

### 4.4 调度与仿真

- Gymnasium：强化学习环境接口
- Stable-Baselines3：首版 PPO 训练
- STK 或可复现的轨道链路生成器：卫星可见性和 SNR 轨迹
- TensorBoard：训练日志

### 4.5 测试与实验管理

- pytest：模块与环境测试
- YAML：实验配置
- CSV / Parquet：中间数据和评估结果
- Matplotlib / Seaborn：论文图表

---

## 5. 核心模块

### 5.1 DOTA 数据预处理与 ROI 提取

职责：

- 解析 DOTA `labelTxt` 旋转四边形标注；
- 对 DOTA 大图进行切片和坐标映射；
- 从标注或检测结果中提取目标；
- 将局部检测结果恢复至原图全局坐标；
- 对重叠切片中的重复目标执行旋转框 NMS；
- 输出统一的 ROI 元数据。


统一 ROI 元数据应至少包含：

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
```

### 5.2 轻量级 ROI 检测器

计划使用 `YOLO11n-obb` 作为源卫星轻量 ROI 提取器。

检测器负责输出：

- 旋转边界框；
- 预测类别；
- 检测置信度；
- 中心坐标；
- ROI 面积占比。

该检测器以高召回率为首要目标。ROI 一旦漏检，后续调度系统无法补救。正式实验需要报告：

- Recall；
- DOTA mAP；
- 参数量和 FLOPs；
- 单切片推理时延；
- 每时隙产生的 ROI 数量。

### 5.3 ROI 语义价值

每个 ROI 的固有语义价值由类别优先级、检测置信度和面积比例共同决定：

```text
v_k = clip(
    w_class * class_priority(y_k)
    + w_confidence * confidence_k
    + w_area * area_ratio_k,
    0,
    1
)
```

语义价值表示 ROI 的业务重要性，不表示最终处理是否正确。最终任务质量由质量代理函数预测。

### 5.4 多出口精细任务模型

多出口模型用于对 ROI 进行精细识别，并支持按计算预算提前退出。

典型结构：

```text
Backbone Stage 1 -> Exit 1
Backbone Stage 2 -> Exit 2
Backbone Stage 3 -> Exit 3
Backbone Stage 4 -> Exit 4
```

每个出口必须记录：

- 任务质量；
- 推理时延；
- 计算量或 CPU/GPU cycles；
- 模型输出置信度。

训练时必须将原始 logits 传入交叉熵损失，不能将 Softmax 概率传入 `CrossEntropyLoss`。

### 5.5 压缩与离线 Profiling

论文方案取消了模型中间切分，因此：

- 本地处理：ROI 不需要通信压缩；
- 跨星卸载：压缩原始 ROI，目标卫星解码后从模型入口执行至指定出口。

建议压缩等级：

```text
Beta 0: 无损或近无损
Beta 1: 高质量压缩
Beta 2: 中等质量压缩与适度缩放
Beta 3: 强压缩与强缩放
```

Profiling 阶段遍历：

```text
ROI x 早退出口 x 压缩等级
```

并记录：

```text
ROI 元特征
出口等级
压缩等级
压缩后字节数
编码/解码时延
模型推理时延
真实任务质量
```

### 5.6 任务质量代理

质量代理用于在线预测某个 ROI 在指定出口和压缩等级下的预期任务质量：

```text
q_k = Proxy(ROI metadata, exit_level, compression_level)
```

首版采用轻量 MLP。输入特征转换器、标准化参数、类别编码顺序和动作编码顺序必须与模型权重共同保存。

代理模型需要在独立测试集上报告：

- MAE；
- MSE；
- R-squared；
- 不同类别、出口和压缩等级下的分组误差。

### 5.7 星间链路与 AMC

链路模块负责生成或读取每个时隙的：

- 协作卫星可见性；
- SNR；
- 传播时延；
- 可用带宽。

系统采用固定发射功率和 SNR-AMC 查表，不在每个 RL step 内求解连续功率优化。

通信时延：

```text
throughput = bandwidth * spectral_efficiency(SNR)
transmission_delay = compressed_size / throughput
```

不可见链路必须标记为非法动作，不能仅使用普通的大时延代替。

### 5.8 计算、队列与能耗

每颗卫星维护独立计算队列。队列按固定计算频率演化：

```text
Q_i(t + 1) = max(Q_i(t) - f_i * delta_t, 0) + arrival_cycles_i(t)
```

任务总时延需要包括：

- 等待时延；
- 编码和解码时延；
- 通信时延；
- 传播时延；
- 模型推理时延；
- 轻量结果回传时延。

能耗包括：

- 固定频率下的计算能耗；
- 固定发射功率下的通信能耗。

### 5.9 网格化 AoSI

DOTA 图像切片仅用于检测计算，不能直接作为 AoSI 网格。

正确流程：

```text
重叠切片检测
    -> 恢复原图全局坐标
    -> 全局旋转框去重
    -> 根据目标中心点分配至固定 AoSI 网格
    -> 跨时隙更新 AoSI
```

网格综合语义价值：

```text
V_g = 1 - product(1 - v_k), k belongs to grid g
```

语义变化因子：

```text
C_g = alpha + (1 - alpha) * abs(V_g(t) - V_g(t - 1))
```

最终 AoSI 代价由网格价值、变化程度和网格年龄共同确定。

DOTA 本身不是连续时序数据集。DOTA 用于视觉质量建模和 ROI 业务流生成；跨时隙变化需要在调度仿真中可控构造，并在实验说明中明确。

### 5.10 强化学习调度环境

首版使用单智能体 PPO 验证完整因果闭环。

状态至少包含：

- ROI 类别、置信度、面积、语义价值和所属网格；
- 各协作卫星的可见性和 SNR；
- 各卫星计算队列；
- 网格价值和 AoSI；
- 质量与时延虚拟队列。

动作至少包含：

```text
丢弃
或
目标节点 x 早退出口 x 压缩等级
```

约束：

- 本地处理固定为无通信压缩；
- 卸载只能选择当前可见卫星；
- 每个 ROI 在一个时隙内只能选择一个原子动作。

奖励应同时考虑：

```text
+ 语义价值加权任务质量
- 端到端时延
- 能耗
- AoSI
- 超时、质量不达标和非法动作惩罚
```

在单智能体环境稳定并通过测试后，再评估是否升级为多智能体强化学习（MADRL）。

---

## 6. 重要工程约束

### 6.1 数据划分与泄漏

- 必须按照 DOTA 原始大图 ID 划分训练集、验证集和测试集。
- 禁止先切片或裁剪 ROI，再随机划分数据。
- 同一原图产生的切片和 ROI 必须全部属于同一数据划分。
- Profiling 训练数据和代理测试数据必须严格隔离。

### 6.2 检测切片与 AoSI 网格

- 检测切片用于适配大幅图像和检测器输入尺寸。
- AoSI 网格用于表示固定物理空间的新鲜度。
- 两者必须独立定义。
- 所有 ROI 必须先恢复至原图或统一地理坐标系，再分配至 AoSI 网格。

### 6.3 ROI 去重

- 重叠切片会导致同一目标被多次检测。
- 在计算语义价值和 AoSI 前，必须完成全局旋转框 NMS 或等效去重。
- 单个 ROI 最终只能属于一个 AoSI 网格，默认使用中心点归属规则。

### 6.4 质量定义

- 单个 ROI 的质量分数不能直接称为 mAP。
- 分类实验可使用正确类别概率或正确性指标。
- 检测实验应结合类别正确性与旋转框 IoU。
- 全局 mAP 只用于数据集级最终评估。

### 6.5 因果闭环

强化学习动作必须真实影响：

- 任务质量；
- 压缩后数据大小；
- 通信时延；
- 计算时延；
- 队列状态；
- 能耗；
- AoSI。

禁止使用与动作无关的固定质量分数或全零代理输入。RL 收敛不能替代环境正确性验证。

### 6.6 可复现性

- 所有实验固定并记录随机种子。
- 所有路径、超参数和权重通过配置文件管理。
- 保存完整环境配置、代理特征转换器和模型版本。
- 评估时对所有策略使用相同场景、相同随机种子和相同链路轨迹。

### 6.7 运行效率

- 大规模 Profiling 应支持批处理和断点恢复。
- 在线环境不能在每个 step 中运行完整视觉模型。
- RL 环境应调用已经固化的质量代理、延迟表和计算成本表。
- 不在每个 step 内执行昂贵的非线性优化或模型训练。

---

## 7. 建议代码结构

```text
leo_experiment/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- dataset.yaml
|   |-- detector.yaml
|   |-- multi_exit.yaml
|   |-- proxy.yaml
|   |-- satellite_env.yaml
|   `-- rl.yaml
|-- data/
|   |-- raw/
|   |-- splits/
|   |-- tiles/
|   |-- roi_crops/
|   |-- profiling/
|   `-- link_traces/
|-- src/
|   |-- data/
|   |-- detector/
|   |-- multi_exit/
|   |-- proxy/
|   |-- simulation/
|   |-- envs/
|   |-- policies/
|   `-- utils/
|-- scripts/
|   |-- 01_prepare_dota.py
|   |-- 02_train_detector.py
|   |-- 03_extract_rois.py
|   |-- 04_train_multi_exit.py
|   |-- 05_generate_profiles.py
|   |-- 06_train_proxy.py
|   |-- 07_validate_environment.py
|   |-- 08_train_ppo.py
|   |-- 09_evaluate_policies.py
|   `-- 10_plot_results.py
|-- tests/
|   |-- test_dota_preprocessing.py
|   |-- test_coordinate_restore.py
|   |-- test_proxy_features.py
|   |-- test_delay_model.py
|   |-- test_queue_model.py
|   |-- test_aosi.py
|   `-- test_environment.py
`-- outputs/
    |-- detector/
    |-- multi_exit/
    |-- proxy/
    |-- rl/
    |-- evaluations/
    `-- figures/
```

职责边界：

- `src/`：可复用的核心实现；
- `scripts/`：实验流水线入口，不承载核心算法；
- `tests/`：验证数据处理、数学模型和环境因果关系；
- `configs/`：保存实验参数；
- `outputs/`：保存模型、评估结果和论文图表。

---

## 8. 实验执行顺序

```text
01_prepare_dota
    -> 02_train_detector
    -> 03_extract_rois
    -> 04_train_multi_exit
    -> 05_generate_profiles
    -> 06_train_proxy
    -> 07_validate_environment
    -> 08_train_ppo
    -> 09_evaluate_policies
    -> 10_plot_results
```

推荐按以下阶段实施：

### 阶段一：可信视觉数据

- 完成 DOTA 原图级划分；
- 训练和评估 YOLO11n-OBB；
- 生成真实检测 ROI 和元数据；
- 完成全局坐标恢复和旋转框去重。

### 阶段二：多出口模型与质量代理

- 训练多出口任务模型；
- 实现真实图像压缩；
- 生成 Profiling 数据；
- 在独立测试集上验证质量代理。

### 阶段三：确定性卫星仿真

- 实现 AMC、通信时延、计算时延、队列、能耗和 AoSI；
- 使用手工动作与基线策略验证环境因果关系；
- 确保不同动作产生符合物理直觉的结果。

### 阶段四：强化学习与正式实验

- 训练单智能体 PPO；
- 加入质量和时延虚拟队列；
- 与各类基线进行统一评估；
- 完成消融实验和论文图表。

---

## 9. 基线与评价指标

### 9.1 必须实现的基线

- Random：随机合法动作；
- Local-Deep：全部本地、最深出口；
- Local-Adaptive：全部本地、自适应早退；
- Best-SNR：卸载至当前最高 SNR 可见卫星；
- Semantic-Greedy：按照 ROI 语义价值贪心调度；
- No-AoSI：不考虑 AoSI 的短视策略；
- Proposed-RL：联合质量、时延、能耗与 AoSI 的强化学习策略。

### 9.2 核心评价指标

- DOTA 检测 mAP 和 ROI Recall；
- 多出口准确率或检测质量；
- 质量代理 MAE、MSE 和 R-squared；
- 高价值 ROI 处理成功率；
- 平均与 P95 端到端时延；
- 超时率和质量约束违背率；
- 通信数据量；
- 计算与通信能耗；
- 平均 AoSI 和高价值网格 AoSI；
- 平均队列长度；
- 在线调度决策耗时。

---

## 10. 项目原则

1. 先建立可信的数据和物理因果关系，再训练强化学习。
2. 将视觉质量、通信代价和计算代价分别验证，避免问题相互掩盖。
3. 在线环境使用代理模型和查表结果，不运行昂贵视觉模型。
4. 强化学习曲线收敛不等于实验有效，环境必须通过单元测试和因果验证。
5. 所有论文结果必须来源于独立测试集、统一场景和可复现实验配置。
