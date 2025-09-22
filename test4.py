import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# --------------------------
# 1. 数据准备模块
# --------------------------
class ConcreteRheologyDataset(Dataset):
    """改进的数据加载类，支持动态序列长度和多目标输出"""

    def __init__(self, csv_path, data_root, seq_length=9, transform=None):
        """
        Args:
            csv_path (str): 标签CSV文件路径
            data_root (str): 图像根目录
            seq_length (int): 序列长度
            transform (callable): 数据增强
        """
        self.df = pd.read_csv(csv_path)
        # 选择要归一化的列（例如 "feature_column"）
        column_to_scale = self.df[["YS"]]  #

        # 创建并拟合归一化器
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 默认[0,1]
        scaled_values = self.scaler.fit_transform(column_to_scale)

        # 将结果存回 DataFrame（新列或覆盖原列）
        self.df["YS_normalized"] = scaled_values

        self.root = data_root
        self.seq_len = seq_length
        self.transform = transform

        # 按组号组织数据
        self.groups = self.df.groupby(['YS','PV','组号'])
        self.valid_groups = []

        for g,path_imgs in self.groups:
            count = 0
            if len(path_imgs) ==9:
                for _, row in path_imgs.iloc[:self.seq_len].iterrows():
                    filename = row['文件名']
                    name = filename.split('\\')
                    f = '/'.join(name)
                    img_path = os.path.join(self.root, f)
                    flag = os.path.exists(img_path)

                    if not flag:
                        break
                    count+=1
                if count ==9:
                    self.valid_groups.append(g)
        # 统计量用于标准化

        self.ys_mean, self.ys_std = self.df['YS'].mean(), self.df['YS'].std()
        self.pv_mean, self.pv_std = self.df['PV'].mean(), self.df['PV'].std()

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        group_id = self.valid_groups[idx]
        group_data = self.groups.get_group(group_id).sort_values('序号')

        # 取固定长度序列
        frames = []
        for _, row in group_data.iloc[:self.seq_len].iterrows():
            filename = row['文件名']
            name =  filename.split('\\')
            f = '/'.join(name)
            img_path = os.path.join(self.root, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise FileNotFoundError(f"图像加载失败: {img_path}")

            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # 标准化标签
        ys = group_data['YS'].iloc[0]
        # ys = group_data['YS_normalized'].iloc[0]
        pv = group_data['PV'].iloc[0]

        return torch.stack(frames), torch.tensor([ys, pv], dtype=torch.float32)


# 数据增强管道
def create_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.225])
    ])


# --------------------------
# 2. 模型架构
# --------------------------
class RheologyPredictor(nn.Module):
    """增强的CNN-LSTM多目标预测模型"""

    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(0.7),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.7),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        # LSTM时序处理
        self.lstm = nn.LSTM(
            input_size=128 * 6 * 6,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # 多任务输出头
        self.ys_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.pv_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, H, W)
        batch_size, seq_len = x.size(0), x.size(1)

        # CNN处理每帧
        cnn_features = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t])  # (B, C, H, W)
            cnn_features.append(feat.view(batch_size, -1))  # (B, D)

        # LSTM处理序列
        lstm_input = torch.stack(cnn_features, dim=1)  # (B, T, D)
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, 2*128)

        # 多目标预测
        ys = self.ys_head(lstm_out[:, -1])  # 取最后时间步
        pv = self.pv_head(lstm_out[:, -1])

        return torch.cat([ys, pv], dim=1)  # (B, 2)


# --------------------------
# 3. 训练与评估工具
# --------------------------
class MetricTracker:
    """指标计算器"""

    def __init__(self):
        self.ys_pred = []
        self.ys_true = []
        self.pv_pred = []
        self.pv_true = []

    def safe_r2_score(self,ys_true, ys_pred):
        """
        安全计算 R² 分数，处理真实值完全相同的特殊情况
        """
        # 计算总平方和 (SST)
        sst = np.sum((ys_true - np.mean(ys_true)) ** 2)
        sse = np.sum((ys_true - ys_pred) ** 2)
        print('sse:',sse)
        # 如果所有真实值相同（SST=0）1-sse/sst
        if sst == 0:

            # 如果预测值也完全相同且等于真实值
            if sse < 0.001:
                return 1.0  # 完美预测
            else:
                return 0.0  # 预测值相同但与真实值不同


        # 正常情况计算 R²
        sse = np.sum((ys_true - ys_pred) ** 2)
        return 1.0 - (sse / sst)
    def update(self, pred, target):
        self.ys_pred.extend(pred[:, 0].cpu().numpy())
        self.pv_pred.extend(pred[:, 1].cpu().numpy())
        self.ys_true.extend(target[:, 0].cpu().numpy())
        self.pv_true.extend(target[:, 1].cpu().numpy())

    def compute(self):
        ys_pred, ys_true = np.array(self.ys_pred), np.array(self.ys_true)
        pv_pred, pv_true = np.array(self.pv_pred), np.array(self.pv_true)

        # MAE计算
        ys_mae = np.mean(np.abs(ys_pred - ys_true))
        pv_mae = np.mean(np.abs(pv_pred - pv_true))

        # R²计算
        ys_r2 = self.safe_r2_score(ys_true,ys_pred)
        pv_r2 = self.safe_r2_score(pv_true,pv_pred)
        t = np.array([ys_true,pv_true])
        p = np.array([ys_pred,pv_pred])
        r2_weighted = r2_score(t, p, multioutput="variance_weighted")
        return {
            'YS_MAE': ys_mae,
            'PV_MAE': pv_mae,
            'YS_R2': ys_r2,
            'PV_R2': pv_r2,
            'r2_weighted':r2_weighted,
        }


def train_model(config):
    """完整的训练流程"""
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    full_dataset = ConcreteRheologyDataset(
        csv_path='dataset_labels.csv',
        data_root='sequences',
        transform=create_transform(),
        seq_length=9
    )

    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset.valid_groups)):
        print(f"\n=== Fold {fold + 1} ===")

        # 划分训练验证集
        val_idx = train_idx[:int(0.2 * len(train_idx))]
        train_idx = train_idx[int(0.2 * len(train_idx)):]

        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, val_idx)
        test_set = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])
        test_loader = DataLoader(test_set, batch_size=config['batch_size'])

        # 模型初始化
        model = RheologyPredictor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        criterion = nn.MSELoss()

        # 训练循环
        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(config['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            for seq, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                seq, target = seq.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(device), target.to(device)
                    output = model(seq)
                    val_loss += criterion(output, target).item()

            # 学习率调整
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{fold}.pt')
            else:
                early_stop_counter += 1
                if early_stop_counter >= config['early_stop']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # 测试评估
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pt'))
        model.eval()
        metric_tracker = MetricTracker()

        with torch.no_grad():
            for seq, target in test_loader:
                seq, target = seq.to(device), target.to(device)
                output = model(seq)
                metric_tracker.update(output, target)

        fold_metrics = metric_tracker.compute()
        results.append(fold_metrics)
        print(f"\nFold {fold + 1} Results:")
        print(f"YS - MAE: {fold_metrics['YS_MAE']:.4f}, R²: {fold_metrics['YS_R2']:.4f},weight r2:{fold_metrics['r2_weighted']:.4f}")
        print(f"PV - MAE: {fold_metrics['PV_MAE']:.4f}, R²: {fold_metrics['PV_R2']:.4f}")

    # 汇总结果
    final_results = {
        'YS_MAE': np.mean([r['YS_MAE'] for r in results]),
        'PV_MAE': np.mean([r['PV_MAE'] for r in results]),
        'YS_R2': np.mean([r['YS_R2'] for r in results]),
        'PV_R2': np.mean([r['PV_R2'] for r in results]),
        'r2_weighted': np.mean([r['r2_weighted'] for r in results])
    }

    print("\n=== Final Cross-Validation Results ===")
    print(f"Average YS - MAE: {final_results['YS_MAE']:.4f}, R²: {final_results['YS_R2']:.4f}")
    print(f"Average PV - MAE: {final_results['PV_MAE']:.4f}, R²: {final_results['PV_R2']:.4f}")
    print('weight r2:',final_results['r2_weighted'])
    return final_results


# --------------------------
# 4. 主程序
# --------------------------
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 1,
        'lr': 3e-4,
        'epochs': 10,
        'early_stop': 10,
    }

    # 检查数据路径
    assert os.path.exists('dataset_labels.csv'), "CSV标签文件不存在"
    assert os.path.exists('sequences'), "图像文件夹不存在"

    # 运行训练
    results = train_model(config)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.bar(['YS', 'PV'], [results['YS_MAE'], results['PV_MAE']])
    plt.title('Mean Absolute Error')

    plt.subplot(122)
    plt.bar(['YS', 'PV'], [results['YS_R2'], results['PV_R2']])
    plt.ylim(0, 1)
    plt.title('R-squared Score')

    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.show()