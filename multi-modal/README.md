# 使用方式
## 使用流程
1. 确保数据文件已存放在 data/ 目录下，格式为 .npy，例如：
    - polarization_features.npy
    - rcs_features.npy
    - range_profile_features.npy
    - micro_doppler_features.npy
    - polarization_labels.npy
2. 配置参数
    - 修改 config/config.yaml 文件
3. 训练模型
    - ```python main.py```
4. 评估模型
    - 训练完成后，模型会自动保存到 checkpoints/ 目录。
    - 通过配置文件修改为验证数据集后，使用以下命令加载模型并评估: ```python main.py --evaluate```



## 组件说明
1. config/ 目录
- config.yaml: 配置文件，定义模型类型、数据路径、超参数等。
- 作用: 通过修改此文件，用户可灵活切换模型、数据集或超参数，无需修改代码。

2. models/ 目录
- base.py: 模型基类 BaseModel，定义通用接口。
- fcn.py: 单模态模型 FCNModel。
- multi_modal.py: 多模态模型 MultiModalModel。
- 作用: 所有模型通过 @REGISTRY.register("model", "name") 注册，支持动态加载。

3. datasets/ 目录
- base.py: 数据集基类 BaseDataset，定义通用接口。
- 作用: 支持单模态和多模态数据加载，通过注册机制动态组合。
4. trainers/ 目录
- base_trainer.py: 通用训练器 BaseTrainer。
- 作用: 提供统一的训练逻辑，支持单模态和多模态模型。
5. evaluators/ 目录
- evaluator.py: 通用评估器 Evaluator。
- 作用: 提供分类报告、混淆矩阵等评估指标。
6. utils/ 目录
- registry.py: 模块注册器 REGISTRY。
- 作用: 实现模型、数据集等模块的动态注册，支持开闭原则。
7. main.py: 主程序入口
- 作用: 根据配置文件动态构建模型、数据集、训练器和评估器。

## ⚙️ 参数说明

### 1. **模型参数 (`config.model`)**
| 参数名       | 类型     | 必需 | 默认值 | 允许值 | 描述 |
|--------------|----------|------|--------|--------|------|
| `type`       | string   | ✅   | -      | 模型名称 | 指定模型类型，如 `fcn` 或 `multi_modal` |
| `params`     | dict     | ✅   | -      | -      | 模型参数，具体取决于模型类型 |

#### 1.1 `params` 参数（以 `multi_modal` 为例）
| 参数名       | 类型     | 必需 | 默认值 | 允许值 | 描述 |
|--------------|----------|------|--------|--------|------|
| `input_dims` | list[int]| ✅   | -      | -      | 各模态的输入维度 |
| `hidden_dim` | int      | ✅   | 128    | >0     | 隐藏层维度 |
| `num_classes`| int      | ✅   | 10     | >0     | 分类类别数 |

---

### 2. **数据参数 (`config.data`)**
| 参数名       | 类型     | 必需 | 默认值 | 允许值 | 描述 |
|--------------|----------|------|--------|--------|------|
| `paths`      | dict     | ✅   | -      | -      | 各模态的数据路径 |
| `labels`     | string   | ✅   | -      | -      | 标签文件路径 |
| `modalities` | list[str]| ✅   | -      | -      | 使用的模态列表 |

#### 2.1 `paths` 参数示例
```yaml
paths:
  polarization: "data/polarization_features.npy"
  rcs: "data/rcs_features.npy"
```

### 3. **训练参数 (`config.training`)**

| 参数名           | 类型   | 必需 | 默认值 | 允许值 | 描述 |
|------------------|--------|------|--------|--------|------|
| `batch_size`     | int    | ✅   | 32     | >0     | 每批次处理的数据量 |
| `num_epochs`     | int    | ✅   | 50     | >0     | 训练过程中的完整遍历数据集次数 |
| `learning_rate`  | float  | ✅   | 1e-3   | >0     | 学习率，控制权重更新步长 |
| `optimizer`      | string | ❌   | "Adam" | "Adam", "SGD" 等 | 优化算法的选择 |
| `loss_function`  | string | ❌   | "CrossEntropyLoss" | 各种损失函数名称 | 损失函数类型 |
| `shuffle`        | bool   | ❌   | true   | true, false | 是否在每个epoch开始时打乱数据 |
| `validation_split`| float | ❌   | 0.2    | [0.0, 1.0) | 验证集划分比例 |



## 扩展与维护
1. 新增模型
    - 在 models/ 中定义新模型类。
    - 使用 @REGISTRY.register("model", "new_model_name") 注册。
    - 修改 config.yaml 中的 model.type 为 "new_model_name"。
2. 新增数据集
    - 在 datasets/ 中定义新数据集类。
    - 使用 @REGISTRY.register("dataset", "new_dataset_name") 注册。
    - 修改 config.yaml 中的 data.modalities 或 data.paths。
3. 新增模态
    - 在 datasets/ 中实现新模态的数据集类。
    - 在 config.yaml 中添加新模态的路径和配置。
    - 无需修改现有代码，直接通过配置文件启用新模态。

4. 新增训练器和评估器，在对应文件夹定义后，在main.py中替换即可，或者直接修改base_trainer或evaluator实现适用于你的逻辑。
>无需修改任何现有代码，只需通过配置文件和新增模块定义即可扩展功能。