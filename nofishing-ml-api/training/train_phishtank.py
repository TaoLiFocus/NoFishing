# -*- coding: utf-8 -*-
"""
用PhishTank真实数据训练URL分类器
"""
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingDataset(Dataset):
    """Phishing URL dataset"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_model(input_dim: int = 20) -> nn.Module:
    """创建模型架构（与url_classifier.py一致）"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    return Model()


def train_model(
    X_path: str = 'data/X_phishtank.npy',
    y_path: str = 'data/y_phishtank.npy',
    model_save_path: str = 'models/phishing_classifier_phishtank.pt',
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    train_split: float = 0.8
):
    """
    用PhishTank数据训练模型

    Args:
        X_path: 特征文件路径
        y_path: 标签文件路径
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        train_split: 训练集比例
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载数据
    logger.info(f"Loading data from {X_path} and {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)

    logger.info(f"Loaded {len(X)} samples: {sum(y==1)} phishing, {sum(y==0)} legitimate")

    # 创建数据集
    dataset = PhishingDataset(X, y)

    # 分割训练集和测试集
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train set: {train_size}, Validation set: {val_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = create_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 8

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True Positive
        val_fn = 0  # False Negative
        val_fp = 0  # False Positive
        val_tn = 0  # True Negative

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                predictions = (outputs >= 0.5).float()

                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)

                # 混淆矩阵
                for i in range(len(y_batch)):
                    if predictions[i] == 1 and y_batch[i] == 1:
                        val_tp += 1
                    elif predictions[i] == 0 and y_batch[i] == 1:
                        val_fn += 1
                    elif predictions[i] == 1 and y_batch[i] == 0:
                        val_fp += 1
                    else:
                        val_tn += 1

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # 计算验证集指标
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            model_path = Path(model_save_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} - "
            f"val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}"
        )

        # 早停
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info("Training complete!")
    logger.info(f"Best model saved to {model_save_path}")


if __name__ == '__main__':
    train_model(
        epochs=30,
        batch_size=64,
        learning_rate=0.001
    )
