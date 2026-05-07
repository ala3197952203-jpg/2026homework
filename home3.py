import json
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = r"C:\Users\大白\Desktop\工作\神经网络课程\home3"
FILES = [
    "poet.song.40000.json",
    "poet.song.41000.json",
    "poet.song.42000.json",
    "poet.song.43000.json"
]
POEM_TYPE = 'five'       # 选择生成格式：'seven'七言 / 'five'五言
START_WORD = '明月'        # 两字总起词
SEQ_LEN = 8               # 训练序列长度（减小）
EMBEDDING_DIM = 64        # 减小维度
HIDDEN_DIM = 128          # 减小维度
NUM_LAYERS = 1            # 减少层数
BATCH_SIZE = 32           # 减小批次
EPOCHS = 20               # 减少轮次
LEARNING_RATE = 0.002
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess():
    """加载JSON文件并筛选对应格式的诗句"""
    corpus = []
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    
    for file_name in FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"正在处理: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                data = json.load(f)
        
        for poem in data:
            paragraphs = poem.get('paragraphs', [])
            for para in paragraphs:
                # 按标点符号分割句子
                sentences = re.split(r'[，。！？；：、]', para)
                for sent in sentences:
                    # 清洗文本：仅保留汉字
                    clean_sent = pattern.sub('', sent)
                    # 筛选对应字数的句子
                    if POEM_TYPE == 'seven' and len(clean_sent) == 7:
                        corpus.append(clean_sent)
                    elif POEM_TYPE == 'five' and len(clean_sent) == 5:
                        corpus.append(clean_sent)
    
    print(f"✅ 加载完成：共{len(corpus)}条{POEM_TYPE}言训练句子")
    return corpus

def build_vocab(corpus):
    """构建字符→索引映射字典"""
    char_counter = Counter()
    for sent in corpus:
        char_counter.update(sent)
    
    # 添加特殊字符
    vocab = ['<pad>', '<unk>'] + [char for char, _ in char_counter.most_common()]
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, vocab

class PoemDataset(Dataset):
    """自定义数据集：将字符序列转为训练样本"""
    def __init__(self, corpus, char_to_idx, seq_len):
        self.seq_len = seq_len
        self.data = []
        
        # 将所有句子拼接为长序列
        full_seq = []
        for sent in corpus:
            full_seq.extend([char_to_idx.get(char, char_to_idx['<unk>']) for char in sent])
        
        # 生成滑动窗口样本
        for i in range(len(full_seq) - seq_len):
            x = full_seq[i:i+seq_len]
            y = full_seq[i+seq_len]  # 只取下一个字符作为目标
            self.data.append((x, y))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTMModel(nn.Module):
    """字符级LSTM古诗生成模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步输出
        return out, hidden


def train_model(model, dataloader, vocab_size):
    """训练模型并记录Loss"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []
    
    model.to(DEVICE)
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}')
    
 
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), loss_history, marker='o')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('损失值 (Loss)')
    plt.title('LSTM训练损失收敛曲线')
    plt.grid(True)
    plt.show()
    
    return model

def generate_poem(model, char_to_idx, idx_to_char):
    """生成四句绝句，第一句前两个字替换为总起词"""
    model.eval()
    sent_len = 7 if POEM_TYPE == 'seven' else 5
    sentences = []
    
    for _ in range(4):
        # 随机初始化第一个字符
        start_char = idx_to_char[torch.randint(0, len(char_to_idx), (1,)).item()]
        current_sent = [start_char]
        
        hidden = None
        input_seq = torch.tensor([[char_to_idx.get(start_char, char_to_idx['<unk>'])]], 
                               dtype=torch.long).to(DEVICE)
        
        for _ in range(sent_len - 1):
            output, hidden = model(input_seq, hidden)
            prob = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(prob, 1).item()  # 使用随机采样避免重复
            next_char = idx_to_char[next_idx]
            current_sent.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
        
        sentences.append(''.join(current_sent))
    
    # 替换第一句前两个字为总起词
    if len(sentences[0]) >= 2:
        sentences[0] = START_WORD + sentences[0][2:]
    
    # 添加标点
    poem = ''
    for i, sent in enumerate(sentences):
        poem += sent + ('，' if i % 2 == 0 else '。')
    
    return poem


def main():
    # 1. 加载数据
    corpus = load_and_preprocess()
    if not corpus:
        print("❌ 未找到符合条件的诗句，请检查POEM_TYPE设置！")
        return
    
    # 2. 构建字典
    char_to_idx, idx_to_char, vocab = build_vocab(corpus)
    print(f"词汇表大小: {len(vocab)}")
    
    # 3. 准备数据集
    dataset = PoemDataset(corpus, char_to_idx, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 4. 初始化模型
    model = LSTMModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 5. 训练模型
    print("🚀 开始训练模型...")
    model = train_model(model, dataloader, len(vocab))
    
    # 6. 生成古诗
    print("\n📜 生成结果：")
    for i in range(3):  # 生成3个不同版本
        poem = generate_poem(model, char_to_idx, idx_to_char)
        print(f"版本{i+1}: {poem}")

if __name__ == '__main__':
    main()