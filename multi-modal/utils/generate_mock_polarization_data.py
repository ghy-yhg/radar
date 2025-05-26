import numpy as np
import os

# 设置参数
num_samples = 1000          # 样本数量
feature_dim = 100           # 每个样本的特征维度（例如 25 个采样点 × 4 个极化通道）
num_classes = 10            # 分类类别数（如 10 类雷达目标）
output_dir = "data/"        # 保存路径
feature_file = os.path.join(output_dir, "polarization_features.npy")
label_file = os.path.join(output_dir, "polarization_labels.npy")

# 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# 模拟极化特征数据（一维向量）
# 假设极化特征由 HH, VV, HV, VH 等通道组成，这里简化为一维向量
np.random.seed(42)  # 保证可复现
features = np.zeros((num_samples, feature_dim))

# 为每个类别生成具有不同统计特性的特征
for i in range(num_samples):
    class_id = np.random.randint(0, num_classes)
    # 每个类别的特征数据具有不同的均值和标准差，以体现区分性
    mean = 1.0 + class_id * 0.5
    std = 0.2 + class_id * 0.05
    features[i, :] = np.random.normal(loc=mean, scale=std, size=feature_dim)

# 生成标签
labels = np.random.randint(0, num_classes, size=num_samples)

# 保存为 .npy 文件
np.save(feature_file, features)
np.save(label_file, labels)

print(f"✅ 极化特征已保存至: {feature_file}")
print(f"✅ 极化标签已保存至: {label_file}")
print(f"样本数量: {num_samples}, 特征维度: {feature_dim}, 类别数: {num_classes}")