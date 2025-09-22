import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import random
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset

def evaluate_model(model, loader, criterion, target_scaler):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        total_loss = 0.0
        for rf_features, features, targets in loader:
            rf_features, features, targets = rf_features.to(device), features.to(device), targets.to(device)
            outputs = model(rf_features, features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * rf_features.size(0)

            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)


    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    predictions_orig = target_scaler.inverse_transform(predictions)
    actuals_orig = target_scaler.inverse_transform(actuals)

    mae = mean_absolute_error(actuals_orig, predictions_orig)
    rmse = np.sqrt(mean_squared_error(actuals_orig, predictions_orig))
    r2 = r2_score(actuals_orig, predictions_orig)
    mape = np.mean(np.abs((actuals_orig - predictions_orig) / np.maximum(np.abs(actuals_orig), 1e-8))) * 100

    ys_pred = predictions_orig[:,0]
    ys_true = actuals_orig[:, 0]
    pv_pred = predictions_orig[:,1]
    pv_true = actuals_orig[:, 1]

    ys_mae = mean_absolute_error(ys_true, ys_pred)
    pv_mae = mean_absolute_error(pv_true, pv_pred)
    ys_r2 = r2_score(ys_true, ys_pred)
    pv_r2 = r2_score(pv_true, pv_pred)
    return avg_loss, mae, rmse, r2, mape, predictions_orig, actuals_orig,ys_mae,ys_r2,pv_mae,pv_r2
def train_hybrid_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for rf_features, features, targets in train_loader:
            rf_features, features, targets = rf_features.to(device), features.to(device), targets.to(device)


            outputs = model(rf_features, features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * rf_features.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rf_features, features, targets in val_loader:
                rf_features, features, targets = rf_features.to(device), features.to(device), targets.to(device)
                outputs = model(rf_features, features)
                val_loss += criterion(outputs, targets).item() * rf_features.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_hybrid_model.pth')


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')


    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('hybrid_loss_curve.png')
    plt.show()

    return train_losses, val_losses
class RFTransformerModel(nn.Module):
    def __init__(self, input_size, num_trees, embedding_dim=32, num_heads=4, num_layers=2):
        super(RFTransformerModel, self).__init__()


        self.vocab_size = num_trees * 100

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,

        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.feature_branch = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )


        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, rf_features, original_features):#前向推理

        rf_features = torch.clamp(rf_features, 0, self.vocab_size - 1) #【0,477】

        embedded = self.embedding(rf_features)
        transformer_out = self.transformer_encoder(embedded)
        rf_output = transformer_out.mean(dim=1)

        feature_output = self.feature_branch(original_features)

        combined = torch.cat((rf_output, feature_output), dim=1)
        output = self.fusion(combined)

        return output
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
        ys = group_data['YS'].iloc[0]
        pv = group_data['PV'].iloc[0]

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
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
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
        n_estimators=200,
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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset.valid_groups)):
        print(f"\n=== Fold {fold + 1} ===")

        # 划分训练验证集
        # val_idx = train_idx[:int(0.2 * len(train_idx))]
        # train_idx = train_idx[int(0.2 * len(train_idx)):]

        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, test_idx)
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

        regressor=None
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
                    # regressor = nn.Linear(features.size(1), 2).to(device)
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
        feature_scaler = StandardScaler()  # 标准化
        target_scaler = StandardScaler()

        train_features = feature_scaler.fit_transform(train_features)
        train_targets = target_scaler.fit_transform(train_targets)
        test_features = feature_scaler.transform(test_features)
        test_targets = target_scaler.transform(test_targets)
        # 训练随机森林
        print("训练随机森林...")
        rf_model0 = train_rf(train_features, train_targets[:,0])
        rf_model1 = train_rf(train_features, train_targets[:, 1])
        # 保存随机森林模型
        joblib.dump(rf_model0, f'rf_model_fold{fold}.pkl')
        joblib.dump(rf_model1, f'rf_model_fold{fold}.pkl')

        train_leaf_indece = rf_model0.apply(train_features)
        test_leaf_indece = rf_model0.apply(test_features)

        rf_train_tensor = torch.tensor(train_leaf_indece, dtype=torch.long)  # numpy 矩阵 pytorch tensor
        X_train_tensor = torch.tensor(train_features, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_targets, dtype=torch.float32)

        rf_test_tensor = torch.tensor(test_leaf_indece, dtype=torch.long)
        X_test_tensor = torch.tensor(test_features, dtype=torch.float32)
        y_test_tensor = torch.tensor(test_targets, dtype=torch.float32)

        train_dataset = TensorDataset(rf_train_tensor, X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(rf_test_tensor, X_test_tensor, y_test_tensor)
        test_dataset = TensorDataset(rf_test_tensor, X_test_tensor, y_test_tensor)

        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_trees = train_leaf_indece.shape[1]
        input_size = train_features.shape[1]
        model = RFTransformerModel(input_size, num_trees).to(device)

        max_index = np.max(train_leaf_indece)
        model.vocab_size = max_index + 100
        model.embedding = nn.Embedding(
            num_embeddings=model.vocab_size,
            embedding_dim=32
        ).to(device)
        # 评估
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        # 训练模型
        train_losses, val_losses = train_hybrid_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=150
        )

        model.load_state_dict(torch.load('best_hybrid_model.pth'))

        print("\n评估测试集性能...")
        avg_loss, mae, rmse, r2, mape, predictions_orig, actuals_orig, ys_mae, ys_r2, pv_mae, pv_r2 = evaluate_model(
            model, test_loader, criterion, target_scaler
        )

        print('ys_mae, ys_r2, pv_mae, pv_r2:',ys_mae, ys_r2, pv_mae, pv_r2)
        results.append({'YS_MAE':ys_mae,'PV_MAE':pv_mae,'YS_R2':ys_r2,'PV_R2':pv_r2})
        #
        # print(f"\nFold {fold + 1} Results:")
        # print(f"YS - MAE: {fold_metrics['YS_MAE']:.4f}, R²: {fold_metrics['YS_R2']:.4f}")
        # print(f"PV - MAE: {fold_metrics['PV_MAE']:.4f}, R²: {fold_metrics['PV_R2']:.4f}")

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 2,
        'lr': 3e-4,
        'epochs': 1,  # 减少epochs，因为CNN训练更快
        'early_stop': 7,
    }

    # 检查数据路径
    assert os.path.exists('dataset_labels.csv'), "CSV标签文件不存在"
    assert os.path.exists('sequences'), "图像文件夹不存在"

    # 运行训练
    results = train_model(config)

    # # 可视化结果
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