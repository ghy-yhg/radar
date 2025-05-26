import numpy as np
import os

# 参数设置
num_samples = 1000
feature_dim = 100
num_classes = 10
output_dir = "data/"

os.makedirs(output_dir, exist_ok=True)
np.random.seed(42)

# 模拟 目标微动特征（周期性运动）
features = np.zeros((num_samples, feature_dim))
labels = np.zeros(num_samples, dtype=int)

for i in range(num_samples):
    class_id = np.random.randint(0, num_classes)
    labels[i] = class_id

    # 不同类别的微动特征具有不同的频率组合
    freq1 = 0.05 + class_id * 0.01
    freq2 = 0.1 + class_id * 0.02
    amplitude1 = 1.0 + class_id * 0.1
    amplitude2 = 0.5 + class_id * 0.05

    micro_signal = amplitude1 * np.sin(2 * np.pi * freq1 * np.arange(feature_dim)) + \
                   amplitude2 * np.sin(2 * np.pi * freq2 * np.arange(feature_dim)) + \
                   np.random.normal(0, 0.1, feature_dim)

    features[i, :] = micro_signal

# 保存文件
np.save(os.path.join(output_dir, "micro_doppler_features.npy"), features)
np.save(os.path.join(output_dir, "micro_doppler_labels.npy"), labels)

print("✅ 微动特征已保存至: data/micro_doppler_features.npy")
print("✅ 微动标签已保存至: data/micro_doppler_labels.npy")