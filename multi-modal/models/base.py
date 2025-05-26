from utils.registry import REGISTRY
import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

# 注册卷积神经网络模型
@REGISTRY.register("model", "cnn")
class CNNModel(BaseModel):
     def __init__(self, input_length, num_classes=10):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        self.input_length = input_length  # 用于调整输入形状

     def forward(self, x):
        x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        return self.model(x)

# 注册全连接神经网络模型
@REGISTRY.register("model", "fcn")
class FCNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super(FCNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 注册多模态模型
@REGISTRY.register("model", "multi_modal")
class MultiModalModel(BaseModel):
    def __init__(self, input_dims, hidden_dim=128, num_classes=10):
        super(MultiModalModel, self).__init__()
        # 四个特征分支：全连接层（或其他模块）
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dims[0], hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dims[1], hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dims[2], hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dims[3], hidden_dim),
                nn.ReLU()
            )
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, *inputs):
        # inputs: [polarization, rcs, range_profile, micro_doppler]
        branch_outputs = [branch(input) for branch, input in zip(self.branches, inputs)]
        fused = torch.cat(branch_outputs, dim=1)
        return self.fusion(fused)