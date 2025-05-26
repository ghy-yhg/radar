import yaml
from utils.registry import REGISTRY
from models.base import BaseModel
from data.dataset.base import BaseDataset, MultiModalDataset
from trainers.base_trainer import BaseTrainer
from evaluators.evaluator import Evaluator
import torch
import os
import argparse 

def build_model(config):
    model_type = config["model"]["type"]
    model_params = config["model"].get("params", {})
    model_class = REGISTRY.get("model", model_type)
    return model_class(**model_params)

def build_dataset(config):
    dataset_type = "multi_modal" if len(config["data"]["modalities"]) > 1 else config["data"]["modalities"][0]
    dataset_class = REGISTRY.get("dataset", dataset_type)
    if dataset_type == "multi_modal":
        paths = {modality: config["data"]["paths"][modality] for modality in config["data"]["modalities"]}
        label_path = config["data"]["labels"]
        return dataset_class(paths=paths, label_path=label_path)
    else:
        path = config["data"]["paths"][dataset_type]
        label_path = config["data"]["labels"]
        return dataset_class(path=path, label_path=label_path)

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Model Training")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 构建数据集
    dataset = build_dataset(config)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    
    # 获取数据集中的类别数
    if hasattr(dataset, 'labels'):
        num_classes = len(torch.unique(torch.tensor(dataset.labels)))
    else:
        # 如果数据集没有直接提供标签，从第一个批次中获取
        for data in train_loader:
            _, labels = data
            num_classes = len(torch.unique(labels))
            break
    
    # 更新模型参数中的类别数
    if "params" not in config["model"]:
        config["model"]["params"] = {}
    config["model"]["params"]["num_classes"] = num_classes

    # 构建模型
    model = build_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint_path = "checkpoints/best_model.pth"
    if args.evaluate:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"已加载模型权重: {checkpoint_path}")
        else:
            print(f"未找到模型权重文件: {checkpoint_path}")
            return

    if not args.evaluate:
        # 构建训练器
        criterion = torch.nn.CrossEntropyLoss()
        learning_rate = float(config["training"]["learning_rate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        trainer = BaseTrainer(model, train_loader, None, criterion, optimizer, device)
        trainer.train(config["training"]["num_epochs"])
        # 保存模型
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"模型已保存到: {checkpoint_path}")

    # 评估
    evaluator = Evaluator(model, train_loader, device)
    evaluator.evaluate()

if __name__ == "__main__":
    main()