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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib


# --------------------------
# 1. 数据准备模块（保持不变）
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
        ys = (group_data['YS'].iloc[0] - self.ys_mean) / self.ys_std
        pv = (group_data['PV'].iloc[0] - self.pv_mean) / self.pv_std

        return torch.stack(frames), torch.tensor([ys, pv], dtype=torch.float32)


# 数据增强管道（保持不变）
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
# 2. 模型架构 - 修改为CNN特征提取器
# --------------------------
class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器，输出特征向量"""

    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        # 时间维度特征处理
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, H, W)
        batch_size, seq_len = x.size(0), x.size(1)

        # 处理每帧图像
        cnn_features = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t])  # (B, C, H, W)
            feat = feat.view(batch_size, -1)  # (B, D)
            cnn_features.append(feat)

        # 组合时间序列特征
        features = torch.stack(cnn_features, dim=2)  # (B, D, T)
        features = self.temporal_pool(features)  # (B, D, 1)
        features = features.squeeze(2)  # (B, D)

        return features


# --------------------------
# 3. 训练与评估工具 - 修改为CNN+随机森林
# --------------------------
def extract_features(model, dataloader, device):
    """使用CNN提取特征"""
    model.eval()
    features = []
    targets = []

    with torch.no_grad():
        for seq, target in tqdm(dataloader, desc="提取特征"):
            seq = seq.to(device)
            feat = model(seq)
            features.append(feat.cpu().numpy())
            targets.append(target.cpu().numpy())

    return np.vstack(features), np.vstack(targets)


def train_rf(features, targets):
    """训练随机森林回归器"""
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(features, targets)
    return rf


def evaluate_rf(model0,model1 ,features, targets):
    """评估随机森林模型"""
    preds0 = model0.predict(features)
    preds1 = model1.predict(features)
    ys_pred = preds0
    ys_true = targets[:, 0]
    pv_pred = preds1
    pv_true = targets[:, 1]

    ys_mae = mean_absolute_error(ys_true, ys_pred)
    pv_mae = mean_absolute_error(pv_true, pv_pred)
    ys_r2 = r2_score(ys_true, ys_pred)
    pv_r2 = r2_score(pv_true, pv_pred)

    return {
        'YS_MAE': ys_mae,
        'PV_MAE': pv_mae,
        'YS_R2': ys_r2,
        'PV_R2': pv_r2
    }


def train_model(config):
    """修改后的训练流程：CNN特征提取 + 随机森林回归"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    full_dataset = ConcreteRheologyDataset(
        csv_path='dataset_labels.csv',
        data_root='sequences',
        transform=create_transform(),
        seq_length=9
    )

    # K折交叉验证
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
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

        # 初始化CNN特征提取器
        model = CNNFeatureExtractor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()

        # 训练CNN特征提取器
        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(config['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            for seq, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                seq, target = seq.to(device), target.to(device)

                optimizer.zero_grad()
                features = model(seq)

                # 添加一个简单的回归头用于CNN训练
                regressor = nn.Linear(features.size(1), 2).to(device)
                output = regressor(features)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(device), target.to(device)
                    features = model(seq)
                    regressor = nn.Linear(features.size(1), 2).to(device)
                    output = regressor(features)
                    val_loss += criterion(output, target).item()

            avg_val_loss = val_loss / len(val_loader)

            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f'best_cnn_fold{fold}.pt')
            else:
                early_stop_counter += 1
                if early_stop_counter >= config['early_stop']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # 加载最佳CNN模型
        model.load_state_dict(torch.load(f'best_cnn_fold{fold}.pt'))

        # 提取特征
        print("提取训练集特征...")
        train_features, train_targets = extract_features(model, train_loader, device)
        print("提取测试集特征...")
        test_features, test_targets = extract_features(model, test_loader, device)

        # 训练随机森林
        print("训练随机森林...")
        rf_model0 = train_rf(train_features, train_targets[:,0])
        rf_model1 = train_rf(train_features, train_targets[:, 1])
        # 保存随机森林模型
        joblib.dump(rf_model0, f'rf_model_fold{fold}.pkl')
        joblib.dump(rf_model1, f'rf_model_fold{fold}.pkl')
        # 评估
        fold_metrics = evaluate_rf(rf_model0,rf_model1, test_features, test_targets)
        results.append(fold_metrics)

        print(f"\nFold {fold + 1} Results:")
        print(f"YS - MAE: {fold_metrics['YS_MAE']:.4f}, R²: {fold_metrics['YS_R2']:.4f}")
        print(f"PV - MAE: {fold_metrics['PV_MAE']:.4f}, R²: {fold_metrics['PV_R2']:.4f}")

    # 汇总结果
    final_results = {
        'YS_MAE': np.mean([r['YS_MAE'] for r in results]),
        'PV_MAE': np.mean([r['PV_MAE'] for r in results]),
        'YS_R2': np.mean([r['YS_R2'] for r in results]),
        'PV_R2': np.mean([r['PV_R2'] for r in results])
    }

    print("\n=== Final Cross-Validation Results ===")
    print(f"Average YS - MAE: {final_results['YS_MAE']:.4f}, R²: {final_results['YS_R2']:.4f}")
    print(f"Average PV - MAE: {final_results['PV_MAE']:.4f}, R²: {final_results['PV_R2']:.4f}")

    return final_results


# --------------------------
# 4. 主程序
# --------------------------
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 2,
        'lr': 3e-4,
        'epochs': 10,  # 减少epochs，因为CNN训练更快
        'early_stop': 7,
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