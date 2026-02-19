"""
NoFishing Deep Learning Model Training
完整训练流程测试
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 添加项目路径
sys.path.insert(0, 'C:/Users/TaoLi/NoFishing/nofishing-ml-api')

import numpy as np
from datetime import datetime

print("=" * 60)
print("NoFishing - Deep Learning Model Training")
print("=" * 60)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Step 1: 导入必要的模块
print("[Step 1] Importing modules...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Installing PyTorch...")
    os.system('pip install torch -q')

# Step 2: 创建合成数据集
print("\n[Step 2] Creating training dataset...")

class SimplePhishingDataset(Dataset):
    """简化的钓鱼网站数据集"""

    # 钓鱼 URL 模式
    PHISHING_PATTERNS = [
        "http://{ip}/login",
        "http://{brand}-{word}.{tld}/verify",
        "http://@{domain}.{tld}/signin",
        "http://{sub}.{sub}.{sub}.{domain}.{tld}/account",
        "http://{domain}.{tld}/secure-login",
    ]

    # 正常 URL 模式
    LEGITIMATE_PATTERNS = [
        "https://www.{domain}.com/",
        "https://api.{domain}.com/v1/endpoint",
        "https://{sub}.{domain}.com/{path}",
        "https://www.{domain}.com/{path}",
        "https://{domain}.com",
    ]

    TLDS = {
        'phishing': ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq'],
        'legitimate': ['.com', '.org', '.net', '.io', '.edu', '.gov']
    }

    BRANDS = ['apple', 'google', 'microsoft', 'amazon', 'paypal', 'facebook']
    WORDS = ['verify', 'secure', 'login', 'account', 'signin', 'update']
    DOMAINS = ['example', 'service', 'company', 'website', 'test', 'sample']
    SUBDOMAINS = ['api', 'mail', 'portal', 'www', 'app', 'mobile']

    def __init__(self, num_samples=2000):
        self.urls = []
        self.labels = []
        self.features = []

        # 生成钓鱼样本
        for _ in range(num_samples // 2):
            url = self._generate_phishing_url()
            self.urls.append(url)
            self.labels.append(1)

        # 生成正常样本
        for _ in range(num_samples // 2):
            url = self._generate_legitimate_url()
            self.urls.append(url)
            self.labels.append(0)

        # 打乱
        combined = list(zip(self.urls, self.labels))
        np.random.shuffle(combined)
        self.urls, self.labels = zip(*combined)

        # 预提取特征
        print(f"  Extracting features for {len(self.urls)} URLs...")
        self.features = [self._extract_features(url) for url in self.urls]

    def _generate_phishing_url(self):
        """生成钓鱼 URL"""
        import random
        import string

        pattern = np.random.choice(self.PHISHING_PATTERNS)

        # IP 地址
        ip = f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"

        # URL 组件
        brand = np.random.choice(self.BRANDS)
        word = np.random.choice(self.WORDS)
        tld = np.random.choice(self.TLDS['phishing'])
        domain = np.random.choice(self.DOMAINS)
        sub = np.random.choice(self.SUBDOMAINS)
        path = ''.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(5, 20)))

        url = pattern.format(
            ip=ip, brand=brand, word=word, tld=tld,
            domain=domain, sub=sub, path=path
        )

        return url.lower()

    def _generate_legitimate_url(self):
        """生成正常 URL"""
        import random
        import string

        pattern = np.random.choice(self.LEGITIMATE_PATTERNS)

        tld = np.random.choice(self.TLDS['legitimate'])
        domain = np.random.choice(self.DOMAINS)
        sub = np.random.choice(self.SUBDOMAINS)
        path = ''.join(np.random.choice(list(string.ascii_lowercase + '/'), np.random.randint(5, 30)))

        url = pattern.format(tld=tld, domain=domain, sub=sub, path=path)
        return url.lower()

    def _extract_features(self, url):
        """提取 URL 特征（50 维向量）"""
        from urllib.parse import urlparse
        import re

        try:
            parsed = urlparse(url)
        except:
            parsed = urlparse('http://invalid')

        features = np.zeros(50, dtype=np.float32)

        # 基础特征
        features[0] = len(url)
        features[1] = len(parsed.hostname or '')
        features[2] = url.count('.')
        features[3] = url.count('@')
        features[4] = url.count('-')

        # 协议
        features[5] = 1.0 if parsed.scheme == 'https' else 0.0

        # IP 地址
        features[6] = 1.0 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed.hostname or '') else 0.0

        # 子域名数量
        parts = (parsed.hostname or '').split('.')
        features[7] = max(0, len(parts) - 2)

        # 路径深度
        features[8] = (parsed.path or '').count('/')

        # 可疑 TLD
        suspiciou_tlds = ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq']
        features[9] = 1.0 if any((parsed.hostname or '').endswith(tld) for tld in suspiciou_tlds) else 0.0

        # 品牌仿冒
        brands = ['apple', 'google', 'microsoft', 'amazon', 'paypal', 'facebook']
        url_lower = url.lower()
        features[10] = 1.0 if any(b in url_lower for b in brands) else 0.0

        # 可疑关键词
        keywords = ['login', 'signin', 'verify', 'account', 'secure', 'password']
        features[11] = 1.0 if any(k in url_lower for k in keywords) else 0.0

        # 数字字符数
        features[12] = sum(c.isdigit() for c in url)

        # 特殊字符数
        special = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`"'
        features[13] = sum(c in special for c in url)

        # 百分比编码
        features[14] = len(re.findall(r'%[0-9a-fA-F]{2}', url))

        # 非ASCII 字符
        features[15] = 1.0 if any(ord(c) > 127 for c in url) else 0.0

        # 域名中的连字符
        features[16] = 1.0 if '-' in (parsed.hostname or '') else 0.0

        # 端口号
        features[17] = parsed.port or -1

        # 查询参数数量
        features[18] = len(parsed.query.split('&')) if parsed.query else 0

        # 有查询参数
        features[19] = 1.0 if parsed.query else 0.0

        # 路径长度
        features[20] = len(parsed.path or '')

        # 其余特征用启发式评分填充
        heuristic_score = 0.0
        if features[6]: heuristic_score += 0.4  # IP
        if not features[5]: heuristic_score += 0.1  # No HTTPS
        if features[9]: heuristic_score += 0.2  # Suspicious TLD
        if features[10]: heuristic_score += 0.15  # Brand
        if features[11]: heuristic_score += 0.15  # Keywords
        if features[3]: heuristic_score += 0.3  # @ symbol

        features[21] = heuristic_score

        # 填充其余维度为0
        for i in range(22, 50):
            features[i] = 0.0

        return features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]).float(),
            torch.tensor(self.labels[idx]).float()
        )

# Step 3: 定义神经网络模型
print("\n[Step 3] Defining neural network model...")

class PhishingClassifier(nn.Module):
    """钓鱼网站分类器 - PyTorch 神经网络"""

    def __init__(self, input_dim=50, hidden_dim=128):
        super(PhishingClassifier, self).__init__()

        # 特征分支
        self.feature_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_branch(x)
        return self.classifier(x)

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhishingClassifier().to(device)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 4: 创建数据集和数据加载器
print("\n[Step 4] Creating dataset and data loaders...")

full_dataset = SimplePhishingDataset(num_samples=2000)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"  Training samples: {train_size}")
print(f"  Validation samples: {val_size}")

# Step 5: 训练模型
print("\n[Step 5] Training model...")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<12} {'Val Acc':<10}")
print("-" * 60)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 5

for epoch in range(20):  # 训练 20 个 epoch
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = (outputs >= 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # 保存最佳模型
        torch.save(model.state_dict(), 'C:/Users/TaoLi/NoFishing/nofishing-ml-api/models/phishing_classifier.pt')
    else:
        patience_counter += 1

    print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<10.4f} {val_loss:<12.4f} {val_acc:<10.4f}")

    if patience_counter >= max_patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

# Step 6: 评估模型
print("\n[Step 6] Evaluating model...")

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in val_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features).squeeze()
        predictions = (outputs >= 0.5).float()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)
cm = confusion_matrix(all_labels, all_predictions)

print(f"\n  Final Metrics:")
print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# Step 7: 测试推理
print("\n[Step 7] Testing inference...")

test_urls = [
    "https://www.google.com/search",
    "http://apple-verify-account.tk/login",
    "http://192.168.1.1@fake-paypal.ml/signin",
    "https://github.com/user/repo",
]

print("\n  Test URL Predictions:")
for url in test_urls:
    # 提取特征
    dataset = SimplePhishingDataset(num_samples=1)
    features = torch.from_numpy(dataset._extract_features(url)).float().unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(features).squeeze()
        is_phishing = output.item() >= 0.5
        confidence = output.item()

    risk_level = 'CRITICAL' if confidence >= 0.8 else 'HIGH' if confidence >= 0.6 else 'MEDIUM' if confidence >= 0.3 else 'LOW'

    print(f"    URL: {url}")
    print(f"      → Phishing: {is_phishing}, Confidence: {confidence:.3f}, Risk: {risk_level}")

print("\n" + "=" * 60)
print(f"Training Complete! Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model saved to: C:/Users/TaoLi/NoFishing/nofishing-ml-api/models/phishing_classifier.pt")
print("=" * 60)
