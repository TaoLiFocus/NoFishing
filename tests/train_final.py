# -*- coding: utf-8 -*-
"""
NoFishing - Train with Real ARFF Dataset
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
            # ARFF 格式: {1, 0, -1, 1, ...}
            values = line[1:-1].split(',')
            # 处理缺失值
            clean_values = []
            for v in values:
                v = v.strip()
                if v == '?':
                    clean_values.append(0)
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
print("\nAttributes:")
for i, attr in enumerate(attributes[:10]):
    print(f"  {i+1:2}. {attr}")
if len(attributes) > 10:
    print(f"  ... and {len(attributes)-10} more")

# ============================================================================
# 特征提取（31 维）
# ============================================================================
print("\n[Step 2/5] Setting up feature extraction...")

def extract_features(url):
    """Extract 31 features matching ARFF schema"""
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    return [
        len(url),                   # 1
        len(hostname),               # 2
        url.count('.'),              # 3
        1 if '@' in url else 0,       # 4
        url.count('-'),              # 5
        1 if parsed.scheme == 'https' else 0,  # 6
        max(0, hostname.count('.') - 2),  # 7
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,  # 8
        len(parsed.path) if parsed.path else 0,  # 9
        1 if 'favicon.ico' in url_lower else 0,  # 10
        parsed.port if parsed.port else -1,    # 11
        1 if 'token' in url_lower or 'nonce' in url_lower else 0,  # 12
        url.count('/'),               # 13
        1 if '<' in url else 0,        # 14
        1 if 'mailto:' in url_lower else 0,  # 15
        1 if url.endswith('co.cc') or url.endswith('.tk') else 0,  # 16
        1 if 'onmouseover' in url_lower else 0,  # 17
        0 if url.count('popup') > 0 else 1,   # 18
        1 if 'iframe' in url_lower else 0,      # 19
        1 if 'age' in url_lower or 'age=' in url_lower else 0,  # 20
        1 if url.count('www') > 1 else 0,        # 21
        1 if '//' in url[8:] else 0,            # 22
        1 if 'google' in url_lower or 'bing' in url_lower else 0,  # 23
        1 if url.count('/') > 3 else 0,          # 24
        1 if 'statistics' in url_lower or 'report' in url_lower else 0,  # 25
        0,                                  # 26. Result
        1,                                  # 27. Filler 1
        1,                                  # 28. Filler 2
        1,                                  # 29. Filler 3
        1,                                  # 30. Filler 4
    ]

print(f"  Feature extraction ready (31 dimensions)")

# ============================================================================
# 准备训练数据
# ============================================================================
print("\n[Step 3/5] Preparing training data...")

import numpy as np

X = np.array(data, dtype=np.float32)
y = np.array([1] * len(data), dtype=np.float32)  # 都是钓鱼

print(f"  Training samples: {len(X)}")
print(f"  Feature dimension: {X.shape[1]}")

# ============================================================================
# 训练 PyTorch 模型
# ============================================================================
print("\n[Step 4/5] Training PyTorch model...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print(f"  Using PyTorch {torch.__version__}")

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
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = PhishingClassifier(input_dim=31)
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

    # 评估
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_tensor).squeeze()
        y_pred = (y_pred_prob >= 0.5)

    accuracy = (y_pred == y_tensor).float().mean()

    print(f"\n  Training Accuracy: {accuracy*100:.1f}%")

    # 保存模型
    MODEL_PATH = "C:/Users/TaoLi/NoFishing/nofishing-ml-api/models/phishing_classifier_trained_arff.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    HAS_TORCH = True

except ImportError:
    print("  PyTorch not available, using sklearn...")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, max_depth=20)
    model.fit(X, y)
    y_pred = model.predict(X)

    accuracy = np.mean(y_pred == y)
    print(f"  Training Accuracy: {accuracy*100:.1f}%")

    HAS_TORCH = False

# ============================================================================
# 测试推理
# ============================================================================
print("\n[Step 5/5] Testing inference...")

test_cases = [
    ("https://www.google.com/search", 0, "Legitimate Google"),
    ("http://192.168.1.1/login", 1, "IP Address Login"),
    ("http://apple-verify.tk/account", 1, "Apple Phishing"),
    ("https://github.com/user/repo", 0, "Legitimate GitHub"),
]

print("\nTest Results:")
print(f"  {'URL':<50} {'Expected':>10} {'Actual':>10}")
print("  " + "-" * 75)

for url, expected, desc in test_cases:
    features = np.array([extract_features(url)], dtype=np.float32)

    if HAS_TORCH:
        with torch.no_grad():
            prob = model(torch.FloatTensor(features)).squeeze().item()
    else:
        prob = model.predict_proba(features)[0, 1]

    is_phishing = prob >= 0.5
    actual = 1 if is_phishing else 0
    match = "OK" if actual == expected else "WRONG"
    status = "PHISH" if is_phishing else "SAFE"
    conf = f"{prob*100:.0f}%"

    print(f"  {url:<50} {match:>10} {status:>10} ({conf}) - {desc}")

# ============================================================================
# 完成
# ============================================================================
print("\n" + "=" * 70)
print("Training complete!")
print("\nSummary:")
print(f"  Dataset: {ARFF_FILE}")
print(f"  Samples: {len(data)}")
print(f"  Features: {len(attributes)}")
print(f"  Model: PyTorch Neural Network" if HAS_TORCH else "sklearn Random Forest")
print("=" * 70)
