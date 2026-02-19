# -*- coding: utf-8 -*-
"""
NoFishing - Train with Real Phishing Dataset
使用真实钓鱼网站数据集训练模型
"""
import sys
import os
import re
from urllib.parse import urlparse

print("=" * 70)
print("NoFishing - Real Dataset Training")
print("=" * 70)

# ============================================================================
# 加载 ARFF 数据集
# ============================================================================
print("\n[Step 1/5] Loading ARFF dataset...")

ARFF_FILE = "D:/BrowserDownload/FishingSource/Training Dataset.arff"

def load_arff(filepath):
    """Load ARFF format file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析属性
    attributes = []
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()

        if line.lower().startswith('@attribute'):
            # 提取属性名
            attr = line.split()[1]
            attributes.append(attr)
        elif line.lower().startswith('@data'):
            data_start = i + 1
            break

    # 解析数据
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('%'):
            # ARFF 格式: {1, 0, -1, 1, ...} 或 {1, 0, ?, 1, ...}
            values = line[1:-1].split(',')
            # 处理缺失值 '?' - 设为 0
            clean_values = []
            for v in values:
                v = v.strip()
                if v == '?':
                    clean_values.append(0)  # 缺失值设为0
                else:
                    try:
                        clean_values.append(int(v))
                    except ValueError:
                        clean_values.append(0)
            data.append(clean_values)

    return data, attributes

data, attributes = load_arff(ARFF_FILE)

print(f"  Loaded {len(data)} samples")
print(f"  Features: {len(attributes)}")

# 显示属性
print("\n  Features:")
for i, attr in enumerate(attributes[:10]):
    print(f"    {i+1:2}. {attr}")
if len(attributes) > 10:
    print(f"    ... and {len(attributes)-10} more")

# ============================================================================
# 特征提取（URL 需要从其他来源获取）
# ============================================================================
print("\n[Step 2/5] Setting up feature extraction...")

def extract_url_features(url):
    """Extract features from URL (22 features)"""
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    features = [
        len(url),
        len(hostname),
        url.count('.'),
        1 if '@' in url else 0,
        url.count('-'),
        1 if parsed.scheme == 'https' else 0,
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,
        max(0, hostname.count('.') - 2),
        (parsed.path or '').count('/'),
        1 if parsed.query else 0,
        1 if hostname.endswith('.xyz') else 0,
        1 if hostname.endswith('.tk') else 0,
        1 if hostname.endswith('.ml') else 0,
        1 if hostname.endswith('.ga') else 0,
        1 if 'apple' in url_lower else 0,
        1 if 'google' in url_lower else 0,
        1 if 'paypal' in url_lower else 0,
        1 if 'login' in url_lower else 0,
        1 if 'verify' in url_lower else 0,
        1 if 'account' in url_lower else 0,
        1 if 'secure' in url_lower else 0,
        sum(c.isdigit() for c in url),
        sum(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`"' for c in url),
    ]
    return features

print("  Feature extraction ready (22 dimensions)")

# ============================================================================
# 使用 ARFF 数据训练
# ============================================================================
print("\n[Step 3/5] Training with ARFF data...")

import numpy as np

# 准备数据
X = np.array(data, dtype=np.float32)
y = np.array([1] * len(data))  # 都是钓鱼网站

print(f"  Training samples: {len(X)}")
print(f"  Feature dimension: {X.shape[1]}")

# 训练 PyTorch 模型
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print("  Using PyTorch...")

    # 转换为张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # 定义模型
    class PhishingClassifier(nn.Module):
        def __init__(self, input_dim=31):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = PhishingClassifier(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    print("  Training complete!")

    # 保存模型
    MODEL_PATH = "C:/Users/TaoLi/NoFishing/nofishing-ml-api/models/phishing_classifier_trained.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    # 测试推理
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_tensor).squeeze().numpy()

    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y)

    print(f"  Training accuracy: {accuracy*100:.1f}%")

    HAS_TORCH = True

except ImportError:
    print("  PyTorch not available, using sklearn...")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, max_depth=20)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    print(f"  Training accuracy: {accuracy*100:.1f}%")
    HAS_TORCH = False

# ============================================================================
# 测试推理
# ============================================================================
print("\n[Step 4/5] Testing inference...")

test_urls = [
    "https://www.google.com/search",
    "http://192.168.1.1/login",
    "http://apple-verify.tk/account",
    "http://paypal-secure.ml/signin",
    "https://github.com/user/repo",
]

print("\n  Test Results:")
print(f"  {'URL':<50} {'Phishing':>10}")
print("  " + "-" * 65)

for url in test_urls:
    features = np.array([extract_url_features(url)], dtype=np.float32)

    if HAS_TORCH:
        with torch.no_grad():
            prob = model(torch.FloatTensor(features)).squeeze().item()
    else:
        prob = model.predict_proba(features)[0, 1]

    is_phishing = prob >= 0.5
    status = "YES" if is_phishing else "NO "
    conf = f"{prob*100:.0f}%"

    print(f"  {url:<50} {status:>10} ({conf})")

# ============================================================================
# 完成
# ============================================================================
print("\n[Step 5/5] Training complete!")
print("\n" + "=" * 70)
print("Summary:")
print(f"  Dataset: {ARFF_FILE}")
print(f"  Samples: {len(data)}")
print(f"  Features: {len(attributes)}")
print(f"  Model: PyTorch Neural Network" if HAS_TORCH else "sklearn Random Forest")
print(f"  Accuracy: {accuracy*100:.1f}%")
print("=" * 70)
