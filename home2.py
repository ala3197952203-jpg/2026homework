import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from scipy.io import loadmat
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 随机种
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据集路径
data_dir = r"C:\Users\大白\Desktop\工作\神经网络课程\home2"
train_file = "train_32x32.mat"
test_file = "test_32x32.mat"

train_path = os.path.join(data_dir, train_file)
test_path = os.path.join(data_dir, test_file)

# 检查数据集
if not os.path.exists(train_path):
    print(f"错误: 训练集文件不存在: {train_path}")
    print("请确保文件路径正确，或从以下地址下载：")
    print("http://ufldl.stanford.edu/housenumbers/")
    print("需要下载的文件: train_32x32.mat 和 test_32x32.mat")
    exit(1)

if not os.path.exists(test_path):
    print(f"错误: 测试集文件不存在: {test_path}")
    exit(1)

# 自定义SVHN数据集类
class SVHNDataset(Dataset):
    def __init__(self, file_path, transform=None, is_train=True):
        
        data = loadmat(file_path)
                
        images = data['X']
        labels = data['y'].flatten() - 1  
        
        images = np.transpose(images, (3, 2, 0, 1))
        
        self.images = images.astype(np.float32) / 255.0  # 归一化到[0, 1]
        self.labels = labels.astype(np.int64)
        
        self.labels[self.labels == 10] = 0
        
        self.transform = transform
        self.is_train = is_train
        
        print(f"数据集 {file_path} 加载成功:")
        print(f"  图像形状: {self.images.shape}")
        print(f"  标签形状: {self.labels.shape}")
        print(f"  类别分布: {np.bincount(self.labels)}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

basic_transform = transforms.Compose([
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], 
                        std=[0.1980, 0.2010, 0.1970])
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], 
                        std=[0.1980, 0.2010, 0.1970])
])

# 创建数据集
print("=" * 50)
print("加载数据集...")
try:
    train_dataset = SVHNDataset(train_path, transform=augment_transform, is_train=True)
    test_dataset = SVHNDataset(test_path, transform=basic_transform, is_train=False)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = basic_transform
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    
except Exception as e:
    print(f"加载数据集时出错: {e}")
    print("请确保.mat文件格式正确,或检查文件是否损坏")
    exit(1)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 定义CNN模型
class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        
        # 卷积块1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 卷积块2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 卷积块3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.25)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 卷积块2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 卷积块3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 全连接层
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="训练", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

# 评估函数
def evaluate(model, data_loader, criterion, device, mode="验证"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=mode, leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    eval_loss = running_loss / len(data_loader)
    eval_acc = 100. * correct / total
    
    return eval_loss, eval_acc

# 主训练循环
def main():
    print("=" * 50)
    print("开始训练...")
    
    # 创建模型
    model = SVHN_CNN(num_classes=10).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        print("使用无verbose参数的学习率调度器")
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rates': []
    }
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0.0
    patience_counter = 0
    patience = 8
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "验证")
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # 学习率调整
        old_lr = current_lr
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"学习率从 {old_lr:.6f} 调整为 {new_lr:.6f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"未提升，连续 {patience_counter} 个epoch验证准确率未提升")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"! 验证准确率在{patience}个epoch内未提升,提前停止训练")
            break
    
    # 加载最佳模型
    print("\n" + "=" * 30)
    print("加载最佳模型进行测试...")
    
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载epoch {checkpoint['epoch']}的最佳模型")
        print(f"最佳验证准确率: {checkpoint['val_acc']:.2f}%")
    else:
        print("未找到保存的最佳模型，使用当前模型")
    
    # 在测试集上评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "测试")
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"\n最终测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    
    return history, model, test_acc

# 修复图表乱码的优化可视化函数
def plot_training_history_fixed(history, test_accuracy):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = list(range(1, len(history['train_acc']) + 1))
    
    # 1. 左上图：训练和验证准确率曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='训练准确率', marker='o', markersize=4)
    ax1.plot(epochs, history['val_acc'], 'orange', linewidth=2, label='验证准确率', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)  
    ax1.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 100])
    
    # 设置合理的x轴范围，避免过长的Epoch
    if len(epochs) > 0:
        ax1.set_xlim([1, len(epochs)])
    
    # 标记最佳验证准确率
    if history['val_acc']:
        best_val_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
        best_val_acc = max(history['val_acc'])
        ax1.axvline(x=best_val_epoch, color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax1.text(best_val_epoch, best_val_acc, f'最佳: {best_val_acc:.2f}%', 
                fontsize=9, color='red', ha='center', va='bottom')
    
    # 2. 右上图：训练和验证损失曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='训练损失', marker='o', markersize=4)
    ax2.plot(epochs, history['val_loss'], 'orange', linewidth=2, label='验证损失', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('损失', fontsize=12)  
    ax2.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 设置合理的x轴范围
    if len(epochs) > 0:
        ax2.set_xlim([1, len(epochs)])
    
    # 3. 左下图：测试准确率柱状图
    ax3 = axes[1, 0]
    bars = ax3.bar(['测试集'], [test_accuracy], color='skyblue', alpha=0.8, width=0.6)
    ax3.set_ylabel('准确率 (%)', fontsize=12)  
    ax3.set_title(f'测试准确率: {test_accuracy:.2f}%', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加网格线
    ax3.set_axisbelow(True)
    
    # 4. 右下图：学习率变化曲线
    ax4 = axes[1, 1]
    if history['learning_rates']:
        ax4.plot(epochs, history['learning_rates'], 'goldenrod', linewidth=2, marker='D', markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('学习率', fontsize=12)  
        ax4.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_yscale('log')
        
        # 标记学习率变化点
        lr_changes = []
        for i in range(1, len(history['learning_rates'])):
            if history['learning_rates'][i] != history['learning_rates'][i-1]:
                lr_changes.append(i+1)  # epoch是从1开始的
        
        for change_epoch in lr_changes:
            if change_epoch <= len(epochs):
                ax4.axvline(x=change_epoch, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('training_history_fixed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已保存修复后的训练历史图表: training_history_fixed.png")

# 可视化一些预测结果
def visualize_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes[:num_samples]):
        img = np.transpose(images[idx], (1, 2, 0))
        
        mean = np.array([0.4377, 0.4438, 0.4728])
        std = np.array([0.1980, 0.2010, 0.1970])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'真实: {labels[idx]}, 预测: {predicted[idx]}', 
                    color='green' if labels[idx] == predicted[idx] else 'red',
                    fontsize=12)
        ax.axis('off')
    
    plt.suptitle('模型预测结果示例（绿色正确，红色错误）', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('prediction_examples_fixed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已保存预测示例图表: prediction_examples_fixed.png")

# 计算并打印混淆矩阵
def plot_confusion_matrix(model, test_loader, device, num_classes=10):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="计算混淆矩阵"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(num_classes)],
                yticklabels=[str(i) for i in range(num_classes)])
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已保存混淆矩阵图表: confusion_matrix.png")
    
    # 计算各类别准确率
    class_accuracy = 100.0 * cm.diagonal() / cm.sum(axis=1)
    print("\n各类别准确率:")
    for i in range(num_classes):
        print(f"类别 {i}: {class_accuracy[i]:.2f}%")

# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("SVHN图像分类任务 - CNN实现")
    print("=" * 50)
    
    # 运行主训练
    try:
        history, model, test_accuracy = main()
        
        # 使用修复后的可视化函数
        plot_training_history_fixed(history, test_accuracy)
        
        # 可视化预测示例
        visualize_predictions(model, test_loader, device)
        
        # 计算并绘制混淆矩阵
        plot_confusion_matrix(model, test_loader, device)
        
        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        print("已保存的文件:")
        print("1. best_model.pth - 最佳模型权重")
        print("2. training_history_fixed.png - 修复后的训练历史图表")
        print("3. prediction_examples_fixed.png - 预测示例图表")
        print("4. confusion_matrix.png - 混淆矩阵图表")
        print("=" * 50)
        
        # 打印模型总结
        print("\n模型架构总结:")
        print(model)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 打印最终性能
        print(f"\n最终性能:")
        print(f"最佳验证准确率: {max(history['val_acc']):.2f}%")
        print(f"最终测试准确率: {test_accuracy:.2f}%")
        print(f"训练轮数: {len(history['train_acc'])}")
        
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n可能的问题和解决方案:")
        print("1. 内存不足: 尝试减小batch_size")
        print("2. 数据集问题: 检查数据集路径和文件格式")