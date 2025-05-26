import numpy as np
import os

# 参数设置
num_samples = 1000
feature_dim = 100
num_classes = 10
output_dir = "data/"

os.makedirs(output_dir, exist_ok=True)
np.random.seed(42)

# 模拟 一维距离像特征（多个高斯峰值）
features = np.zeros((num_samples, feature_dim))
labels = np.zeros(num_samples, dtype=int)

for i in range(num_samples):
    class_id = np.random.randint(0, num_classes)
    labels[i] = class_id

    # 每个类别的距离像具有不同数量的峰值
    num_peaks = 2 + class_id % 4  # 类别 0~9 的峰值数为 2~5
    profile = np.zeros(feature_dim)
    for _ in range(num_peaks):
        center = np.random.randint(10, feature_dim - 10)
        width = np.random.uniform(1, 3)
        height = np.random.uniform(0.5, 2.0)
        profile += height * np.exp(-((np.arange(feature_dim) - center) ** 2) / (2 * width ** 2))

    features[i, :] = profile

# 保存文件
np.save(os.path.join(output_dir, "range_profile_features.npy"), features)
np.save(os.path.join(output_dir, "range_profile_labels.npy"), labels)

print("✅ 一维距离像特征已保存至: data/range_profile_features.npy")
print("✅ 一维距离像标签已保存至: data/range_profile_labels.npy")