import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in self.dataloader:
                inputs, labels = self._prepare_data(data)
                outputs = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        self.plot_confusion_matrix(all_labels, all_preds)

    def _prepare_data(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            inputs, labels = data
        elif isinstance(data, list):
            inputs, labels = data
        else:
            raise ValueError("Invalid data format")
            
        # 处理多模态输入
        if isinstance(inputs, list):
            inputs = [x.to(self.device).float() for x in inputs]  # 确保数据类型为 float32
        else:
            inputs = inputs.to(self.device).float()  # 确保数据类型为 float32
            
        # 处理标签
        labels = labels.to(self.device).long()  # 标签应该是 long 类型
        
        return inputs, labels

    def plot_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()