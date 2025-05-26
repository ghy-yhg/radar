import numpy as np
from torch.utils.data import Dataset
from utils.registry import REGISTRY

class BaseDataset(Dataset):
    """
    数据集基类，定义通用接口
    """
    def __init__(self, path, label_path=None, transform=None):
        """
        :param path: 数据文件路径
        :param label_path: 标签文件路径
        :param transform: 数据转换函数（可选）
        """
        self.path = path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_data(self, path):
        """加载 .npy 文件"""
        return np.load(path)

    def _apply_transform(self, data):
        """应用数据转换"""
        if self.transform is not None:
            return self.transform(data)
        return data


@REGISTRY.register("dataset", "polarization")
class PolarizationDataset(BaseDataset):
    """
    极化特征数据集
    """
    def __init__(self, path, label_path, transform=None):
        super(PolarizationDataset, self).__init__(path, label_path, transform)
        self.features = self._load_data(path)
        self.labels = self._load_data(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = self._apply_transform(feature)
        return feature, label


@REGISTRY.register("dataset", "rcs")
class RCSLoader(BaseDataset):
    """
    RCS 特征数据集
    """
    def __init__(self, path, label_path, transform=None):
        super(RCSLoader, self).__init__(path, label_path, transform)
        self.features = self._load_data(path)
        self.labels = self._load_data(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = self._apply_transform(feature)
        return feature, label


@REGISTRY.register("dataset", "range_profile")
class RangeProfileDataset(BaseDataset):
    """
    一维距离像特征数据集
    """
    def __init__(self, path, label_path, transform=None):
        super(RangeProfileDataset, self).__init__(path, label_path, transform)
        self.features = self._load_data(path)
        self.labels = self._load_data(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = self._apply_transform(feature)
        return feature, label


@REGISTRY.register("dataset", "micro_doppler")
class MicroDopplerDataset(BaseDataset):
    """
    微动特征数据集
    """
    def __init__(self, path, label_path, transform=None):
        super(MicroDopplerDataset, self).__init__(path, label_path, transform)
        self.features = self._load_data(path)
        self.labels = self._load_data(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = self._apply_transform(feature)
        return feature, label


@REGISTRY.register("dataset", "multi_modal")
class MultiModalDataset(Dataset):
    """
    多模态数据集（支持多个特征模态）
    """
    def __init__(self, paths, label_path, transforms=None):
        """
        :param paths: 每个模态的数据路径字典，例如 {"polarization": "...", "rcs": "..."}
        :param label_path: 标签文件路径
        :param transforms: 每个模态的转换函数字典，例如 {"polarization": transform1, ...}
        """
        self.paths = paths
        self.label_path = label_path
        self.transforms = transforms or {}

        # 加载所有模态的特征数据
        self.modalities = list(paths.keys())
        self.features = {modality: np.load(path) for modality, path in paths.items()}
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = []
        for modality in self.modalities:
            feature = self.features[modality][idx]
            if modality in self.transforms:
                feature = self.transforms[modality](feature)
            features.append(feature)
        label = self.labels[idx]
        return features, label