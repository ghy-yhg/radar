import torch
from torch.utils.data import DataLoader

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for data in self.train_loader:
                inputs, labels = self._prepare_data(data)
                outputs = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(self.train_loader):.4f}")

    def _prepare_data(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            inputs, labels = data
        elif isinstance(data, list):
            inputs, labels = data
        else:
            raise ValueError("Invalid data format")
            
        # 处理多模态输入
        if isinstance(inputs, list):
            inputs = [x.to(self.device).float() for x in inputs]
        else:
            inputs = inputs.to(self.device).float()
            
        # 处理标签
        labels = labels.to(self.device).long()
        
        return inputs, labels