from collections import defaultdict

class Registry:
    def __init__(self):
        self._registry = defaultdict(dict)

    def register(self, module_type, name=None):
        def wrapper(cls):
            nonlocal name
            if name is None:
                name = cls.__name__.lower()
            self._registry[module_type][name] = cls
            return cls
        return wrapper

    def get(self, module_type, name):
        return self._registry[module_type].get(name)

# 全局注册器: 用于注册模型
REGISTRY = Registry()