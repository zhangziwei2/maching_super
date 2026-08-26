"""
SAE模型训练脚本 (基于固定50维特征)
读取原始传感器数据，提取50维特征，训练Sparse Autoencoder。

输出: models/sae_model.pth, models/scaler.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 训练数据路径 (命令行参数 > 环境变量 > 默认值)
DATA_BASE = os.environ.get('CHATTER_DATA_DIR', r"D:\senordata\virbrant_clear")

# SAE参数
ENCODING_DIM = 16
INPUT_DIM = 50  # 3传感器×15 + 力×5
EPOCHS = 300
BATCH_SIZE = 128
LR = 0.001
SPARSITY_WEIGHT = 0.1
SPARSITY_TARGET = 0.05
PATIENCE = 70
TOTAL_SAMPLES = 4270
VAL_SAMPLES = 400

# 特征名 (与feature_extractor.py一致)
VIB_FEATURE_NAMES = [
    'Clearance_Factor', 'Power_Spectrum_Clearance', 'Peak', 'Peak_to_Peak',
    'RMS', 'Std', 'STFT_Mean', 'Variance', 'Signal_Energy', 'Spectral_Energy',
    'STFT_Total_Energy', 'Time_Frequency_Entropy', 'Power_Spectrum_Peak',
    'Frequency_Variance', 'Shape_Factor',
]
FORCE_FEATURE_NAMES = [
    'Force_Freq_Variance', 'Force_Peak2Peak', 'Force_Impulse_Factor',
    'Force_Peak', 'Force_Crest_Factor',
]


# ==================== SAE模型 ====================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_weight=0.1, sparsity_target=0.05):
        super().__init__()
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
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

    def forward(self, x):
        return self.encoder(x), self.decoder(self.encoder(x))

    def sparse_loss(self, encoded):
        rho_hat = torch.mean(torch.sigmoid(encoded), dim=0)
        rho = self.sparsity_target
        kl = rho * torch.log(rho / (rho_hat + 1e-10)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        return torch.sum(kl)

    def get_encoded(self, x):
        return self.encoder(x)


# ==================== 数据加载 (复用现有feature Excel) ====================
def load_training_features():
    """
    从现有feature Excel文件中加载特征数据。
    每个Excel文件有28列: 第1列标签 + 27列特征。
    """
    from feature_extractor import extract_vibration_features, extract_force_features

    print("=" * 60)
    print("加载训练数据...")

    # 数据文件路径 (来自 1.2_power_merge.py + keepinline.py 的输出)
    vib_paths = {
        '主轴': os.path.join(DATA_BASE, '提取的特征', '主轴', '1-5', '振动数据_特征提取结果.xlsx'),
        'X': os.path.join(DATA_BASE, '提取的特征', 'X', '1-5', '振动数据_特征提取结果.xlsx'),
        'Y': os.path.join(DATA_BASE, '提取的特征', 'Y', '1-5', '振动数据_特征提取结果.xlsx'),
    }

    # 力传感器路径
    force_path = os.path.join(DATA_BASE, '提取的特征', '三向力_合力', '振动数据_特征提取结果.xlsx')

    all_data = []
    labels = None

    for name, path in vib_paths.items():
        print(f"\n  加载{name}轴振动数据: {path}")
        if not os.path.exists(path):
            print(f"  ⚠️ 文件不存在: {path}")
            print(f"  请确保已运行 1.3_putFeatures.py 生成特征文件")
            sys.exit(1)
        df = pd.read_excel(path).head(TOTAL_SAMPLES)
        if labels is None:
            labels = df.iloc[:, 0].values
        # 第2-28列为27个特征
        data = df.iloc[:, 1:16].values  # 取前15列
        all_data.append(data)
        print(f"  形状: {data.shape}")

    # 力传感器
    print(f"\n  加载力传感器数据: {force_path}")
    if not os.path.exists(force_path):
        print(f"  ⚠️ 文件不存在: {force_path}")
        sys.exit(1)
    df_force = pd.read_excel(force_path).head(TOTAL_SAMPLES)
    # 力特征取第4-8列
    force_data = df_force.iloc[:, 3:8].values
    all_data.append(force_data)
    print(f"  形状: {force_data.shape}")

    X = np.hstack(all_data)
    print(f"\n合并后特征维度: {X.shape} (样本×50)")
    print(f"标签分布: {np.bincount(labels.astype(int))}")

    return X, labels


def load_from_raw_data():
    """
    备选: 从原始信号数据中提取50维特征。
    使用 keepinline.py 输出作为输入。
    """
    from feature_extractor import extract_vibration_features, extract_force_features

    print("=" * 60)
    print("从原始信号数据提取50维特征...")

    # 读取重排后的数据
    base_dir = os.path.join(DATA_BASE, '三向力_合力6_9')

    # 读取三个振动轴和力的原始信号
    vib_dirs = {
        '主轴': os.path.join(DATA_BASE, '数据', '振动数据_主轴.xlsx'),
        'X': os.path.join(DATA_BASE, '数据', '振动数据_X轴.xlsx'),
        'Y': os.path.join(DATA_BASE, '数据', '振动数据_Y轴.xlsx'),
    }

    all_vib_data = {}
    labels = None

    for name, path in vib_dirs.items():
        if not os.path.exists(path):
            print(f"  ⚠️ 文件不存在: {path}")
            print(f"  尝试从特征Excel加载...")
            return None
        df = pd.read_excel(path).head(TOTAL_SAMPLES)
        signal = df.iloc[:, 0].values.astype(float)
        if labels is None:
            labels = df.iloc[:, 1].values if df.shape[1] > 1 else None
        all_vib_data[name] = signal
        print(f"  {name}: {len(signal)} 点")

    # 这里需要按工况分段提取特征 - 简化起见直接使用现有Excel
    print("  直接复用特征Excel文件...")
    return None


# ==================== 训练 ====================
def train_sae(X_train, X_val, y_train, y_val):
    print(f"\n开始训练SAE (编码维度={ENCODING_DIM})")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = SparseAutoencoder(INPUT_DIM, ENCODING_DIM, SPARSITY_WEIGHT, SPARSITY_TARGET).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    best_val = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for (data,) in train_loader:
            data = data.to(DEVICE)
            enc, dec = model(data)
            loss = criterion(dec, data) + model.sparsity_weight * model.sparse_loss(enc)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(DEVICE)
                enc, dec = model(data)
                loss = criterion(dec, data) + model.sparsity_weight * model.sparse_loss(enc)
                val_loss += loss.item()

        train_avg = train_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        scheduler.step(val_avg)
        cur_lr = optimizer.param_groups[0]['lr']

        # 每10 epoch评估分类准确率
        val_acc = 0
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                enc_train = model.get_encoded(X_train_t).cpu().numpy()
                enc_val = model.get_encoded(X_val_t).cpu().numpy()
                clf = SVC(kernel='rbf', C=1.0, gamma='scale')
                clf.fit(enc_train, y_train)
                val_acc = accuracy_score(y_val, clf.predict(enc_val))

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_avg)
        history['val_loss'].append(val_avg)
        history['val_accuracy'].append(val_acc)
        history['lr'].append(cur_lr)

        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f'早停触发于 epoch {epoch + 1}')
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {train_avg:.6f}/{val_avg:.6f} Acc: {val_acc:.4f} LR: {cur_lr:.6f}')

    model.load_state_dict(best_state)
    print(f"\n最佳验证损失: {best_val:.6f}")
    print(f"最高验证准确率: {max(history['val_accuracy']):.4f}")
    return model, history


# ==================== 保存与可视化 ====================
def save_model(model, scaler, info_retention, svm_acc, rf_acc, history):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # SAE模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'encoding_dim': ENCODING_DIM,
        'info_retention': info_retention,
        'svm_accuracy': svm_acc,
        'rf_accuracy': rf_acc,
    }, os.path.join(MODEL_DIR, 'sae_model.pth'))

    # Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    print(f"\n模型已保存至: {MODEL_DIR}")
    print(f"  sae_model.pth")
    print(f"  scaler.pkl")

    # 报告
    report = f"""SAE训练报告
================
输入维度: {INPUT_DIM}
编码维度: {ENCODING_DIM}
信息保留率: {info_retention:.2f}%
SVM准确率: {svm_acc:.4f}
RF准确率: {rf_acc:.4f}
训练轮数: {len(history['train_loss'])}
"""
    with open(os.path.join(MODEL_DIR, 'sae_training_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)


def visualize_training(history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend()

    axes[0, 1].plot(history['epoch'], history['val_accuracy'], 'g-')
    axes[0, 1].set_title('Val Accuracy'); axes[0, 1].set_ylim(0, 1)

    axes[0, 2].plot(history['epoch'], history['lr'])
    axes[0, 2].set_title('Learning Rate')

    axes[1, 0].semilogy(history['epoch'], history['train_loss'], label='Train')
    axes[1, 0].semilogy(history['epoch'], history['val_loss'], label='Val')
    axes[1, 0].set_title('Loss (log)'); axes[1, 0].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: {MODEL_DIR}/training_history.png")


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("SAE模型训练 (固定50维特征)")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 1. 加载数据
    X, labels = load_training_features()

    # 2. 划分训练/验证集
    np.random.seed(42)
    n = X.shape[0]
    val_idx = np.random.choice(n, VAL_SAMPLES, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # 3. 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 4. 训练SAE
    model, history = train_sae(X_train_s, X_val_s, y_train, y_val)

    # 5. 提取编码特征并评估
    with torch.no_grad():
        X_all_s = scaler.transform(X)
        X_all_t = torch.FloatTensor(X_all_s).to(DEVICE)
        X_encoded = model.get_encoded(X_all_t).cpu().numpy()
        X_recon = model(X_all_t)[1].cpu().numpy()

    mse = np.mean((X_all_s - X_recon) ** 2)
    var_orig = np.var(X_all_s)
    info_ret = (1 - mse / var_orig) * 100

    X_enc_train = X_encoded[:len(X_train)]
    X_enc_val = X_encoded[len(X_train):]

    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_enc_train, y_train)
    svm_acc = accuracy_score(y_val, svm.predict(X_enc_val))

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_enc_train, y_train)
    rf_acc = accuracy_score(y_val, rf.predict(X_enc_val))

    print(f"\n信息保留率: {info_ret:.2f}%")
    print(f"SVM准确率: {svm_acc:.4f}")
    print(f"RF准确率: {rf_acc:.4f}")

    # 6. 保存
    save_model(model, scaler, info_ret, svm_acc, rf_acc, history)
    visualize_training(history)

    # 7. 保存编码特征供融合训练使用
    enc_df = pd.DataFrame(X_encoded, columns=[f'SAE_{i+1}' for i in range(ENCODING_DIM)])
    enc_df['label'] = labels
    enc_df.to_csv(os.path.join(MODEL_DIR, 'sae_encoded_features.csv'), index=False)
    print(f"编码特征已保存: {MODEL_DIR}/sae_encoded_features.csv")


if __name__ == '__main__':
    main()
