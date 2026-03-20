import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)

print("1: 读取数据")
file_path = r'C:\Users\大白\Downloads\Concrete_Data_Yeh.csv'
try:
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"数据前5行:\n{df.head()}")
    print(f"列名: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
    print("示例")
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 8)
    y = 25 + np.sum(X * np.random.randn(8), axis=1) + np.random.randn(n_samples) * 2
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(8)])
    df['target'] = y

# 2数据预处理
print("\n2: 数据预处理")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

print(f"处理前的0值数量: {(X == 0).sum()}")
imputer = SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X)
print(f"处理后的0值数量: {(X == 0).sum()}")

# 3特征选择
print("\n3: 特征选择")

# 计特征/相关性
feature_names = df.columns[:-1].tolist()
correlations = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y.flatten())[0, 1]
    correlations.append((feature_names[i], abs(corr)))

# 相关性排序
correlations.sort(key=lambda x: x[1], reverse=True)
print("特征相关性排序:")
for name, corr in correlations:
    print(f"  {name}: {corr:.4f}")

selected_features_indices = [i for i, (name, _) in enumerate(correlations[:6])]
X_selected = X[:, selected_features_indices]
selected_feature_names = [name for name, _ in correlations[:6]]
print(f"\n选择的特征: {selected_feature_names}")

# PCA特征提取
print("\n4: 使用PCA进行特征提取")
pca = PCA(n_components=0.95)  # 保留95%的方差
X_pca = pca.fit_transform(X)
print(f"PCA后特征数: {X_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_}")

X_final = X_pca
print(f"最终特征形状: {X_final.shape}")

# 5. 数据划分和标准化
print("\n5: 数据划分和标准化")

# 前80%训练，后20%测试
split_idx = int(0.8 * len(X_final))
X_train, X_test = X_final[:split_idx], X_final[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义神经网络模型
print("\n6: 构建神经网络模型")

class ConcreteNet(nn.Module):
    def __init__(self, input_dim):
        super(ConcreteNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 初始化
input_dim = X_final.shape[1]
model = ConcreteNet(input_dim)
print(f"模型结构:\n{model}")
print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 训练
print("\n7: 训练神经网络模型...")
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    train_loss = epoch_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        epoch_val_loss = 0
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item() * batch_X.size(0)
        
        val_loss = epoch_val_loss / len(test_loader.dataset)
        val_losses.append(val_loss)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# 测试
print("\n8: 测试模型性能...")
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    # 计算MSE
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    print(f"测试集性能指标:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

# 可视化结果
print("\n9: 可视化结果")

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 损失曲线
ax1 = axes[0, 0]
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 预测vs真实值散点图
ax2 = axes[0, 1]
ax2.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
# 理想预测线
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Predictions vs Actual (R²={r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 残差图
ax3 = axes[0, 2]
residuals = y_pred - y_true
ax3.scatter(y_true, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Actual Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')
ax3.grid(True, alpha=0.3)

# 特征相关性热图
ax4 = axes[1, 0]
corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(feature_names)))
ax4.set_yticks(range(len(feature_names)))
ax4.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(feature_names, fontsize=8)
ax4.set_title('Feature Correlation Heatmap')
plt.colorbar(im, ax=ax4, shrink=0.8)

# 误差分布
ax5 = axes[1, 1]
errors = y_pred - y_true
ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7)
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Prediction Error')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Error Distribution')
ax5.grid(True, alpha=0.3)

# 实际vs预测对比线图
ax6 = axes[1, 2]
sample_indices = np.arange(min(50, len(y_true)))
ax6.plot(sample_indices, y_true[:50], 'o-', label='Actual', linewidth=2, markersize=6)
ax6.plot(sample_indices, y_pred[:50], 's-', label='Predicted', linewidth=2, markersize=6)
ax6.set_xlabel('Sample Index')
ax6.set_ylabel('Target Value')
ax6.set_title('Actual vs Predicted (First 50 Samples)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 保存模型
print("\n10: 保存模型和结果")
try:
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'pca': pca
    }, 'concrete_model.pth')
    print("模型已保存为 'concrete_model.pth'")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten(),
        'Residual': residuals.flatten()
    })
    results_df.to_csv('concrete_predictions.csv', index=False)
    print("预测结果已保存为 'concrete_predictions.csv'")
    
    # 保存性能指标
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Value': [mse, rmse, mae, r2]
    })
    metrics_df.to_csv('concrete_metrics.csv', index=False)
    print("性能指标已保存为 'concrete_metrics.csv'")
    
except Exception as e:
    print(f"保存文件时出错: {e}")

print("\nSuccess")