"""
SAE 稀疏自编码器 — 对齐参考脚本 2.2_SAEDimensionreduction.py

架构：50→512→256→128→16（编码）→128→256→512→50（解码）
训练参数：epochs=300, batch_size=128, lr=0.001, patience=70
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


class SparseAutoencoder(nn.Module):
    """稀疏自编码器 — 架构与 2.2_SAEDimensionreduction.py 完全一致
    编码: input→512→256→128→encoding_dim
    解码: encoding_dim→128→256→512→input
    """

    def __init__(self, input_dim: int, encoding_dim: int = 16,
                 sparsity_weight: float = 0.1, sparsity_target: float = 0.05):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def sparse_loss(self, encoded):
        rho_hat = torch.mean(torch.sigmoid(encoded), dim=0)
        rho = self.sparsity_target
        kl = rho * torch.log(rho / (rho_hat + 1e-10)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        return torch.sum(kl)

    def get_encoded(self, x):
        """仅返回编码特征（推理时使用）"""
        with torch.no_grad():
            return self.encoder(x)


def train_sae(X_train: np.ndarray, X_val: np.ndarray,
              encoding_dim: int = 16, epochs: int = 300,
              batch_size: int = 128, lr: float = 0.001,
              patience: int = 70, device: str = None) -> tuple:
    """
    训练 SAE，返回 (model, history).

    Args:
        X_train: 训练集 (n_samples, 50)
        X_val:   验证集 (n_samples, 50)
        encoding_dim: 编码维度
        epochs: 最大训练轮数
        batch_size: 批次大小
        lr: 学习率
        patience: 早停耐心值
        device: 设备 (auto-detect if None)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"SAE 训练设备: {device}")

    input_dim = X_train.shape[1]

    # 数据
    train_ds = TensorDataset(torch.FloatTensor(X_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 模型
    model = SparseAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5
    )

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_sparse': [], 'val_sparse': [],
    }

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_total, train_recon, train_sparse = 0, 0, 0
        for (data,) in train_loader:
            data = data.to(device)
            encoded, decoded = model(data)
            recon = criterion(decoded, data)
            sparse = model.sparse_loss(encoded)
            loss = recon + model.sparsity_weight * sparse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item()
            train_recon += recon.item()
            train_sparse += sparse.item()

        # 验证
        model.eval()
        val_total, val_recon, val_sparse = 0, 0, 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device)
                encoded, decoded = model(data)
                recon = criterion(decoded, data)
                sparse = model.sparse_loss(encoded)
                loss = recon + model.sparsity_weight * sparse

                val_total += loss.item()
                val_recon += recon.item()
                val_sparse += sparse.item()

        # 平均
        train_avg = train_total / len(train_loader)
        val_avg = val_total / len(val_loader)
        scheduler.step(val_avg)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_avg)
        history['val_loss'].append(val_avg)
        history['train_recon'].append(train_recon / len(train_loader))
        history['val_recon'].append(val_recon / len(val_loader))
        history['train_sparse'].append(train_sparse / len(train_loader))
        history['val_sparse'].append(val_sparse / len(val_loader))

        # 早停
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  SAE 新最佳 Epoch {epoch+1}: val_loss={val_avg:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  SAE 早停 @ Epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  SAE Epoch {epoch+1}/{epochs}: train={train_avg:.6f} val={val_avg:.6f}")

    model.load_state_dict(best_state)
    print(f"SAE 训练完成，最佳 val_loss={best_val_loss:.6f}")
    return model, history


def save_model(model: SparseAutoencoder, save_dir: str, name: str = 'sae_model'):
    """保存 SAE 模型"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.encoder[0].in_features,
        'encoding_dim': model.encoder[-1].out_features,
    }, path)
    print(f"SAE 模型已保存: {path}")
    return path
