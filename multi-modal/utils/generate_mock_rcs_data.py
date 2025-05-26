import numpy as np
import os

# 参数设置
num_samples = 1000
feature_dim = 100
num_classes = 10
output_dir = "data/"

os.makedirs(output_dir, exist_ok=True)
np.random.seed(42)

# 模拟 RCS 特征（雷达散射截面）
features = np.zeros((num_samples, feature_dim))
labels = np.zeros(num_samples, dtype=int)

for i in range(num_samples):
    class_id = np.random.randint(0, num_classes)
    labels[i] = class_id

    # 基线值（不同类别的 RCS 基线不同）
    baseline = 0.5 + class_id * 0.2
    # 添加周期性波动（模拟角度变化）
    freq = 0.02 + class_id * 0.01
    rcs_values = baseline + 0.5 * np.sin(2 * np.pi * freq * np.arange(feature_dim)) + \
                 np.random.normal(0, 0.1, feature_dim)

    features[i, :] = rcs_values

# 保存文件
np.save(os.path.join(output_dir, "rcs_features.npy"), features)
np.save(os.path.join(output_dir, "rcs_labels.npy"), labels)

print("✅ RCS 特征已保存至: data/rcs_features.npy")
print("✅ RCS 标签已保存至: data/rcs_labels.npy")