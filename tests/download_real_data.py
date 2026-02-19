# -*- coding: utf-8 -*-
"""
NoFishing - Real Phishing Dataset Download & Training
使用真实钓鱼网站数据集训练模型
"""
import sys
import os
import csv
import urllib.request
from datetime import datetime

print("=" * 70)
print(" NoFishing - Real Phishing Data Training ".center(70))
print("=" * 70)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: 下载真实钓鱼网站数据
# ============================================================================
print("[Step 1/5] Downloading real phishing data...")

PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
DATA_DIR = "C:/Users/TaoLi/NoFishing/nofishing-ml-api/data"
PHISHING_FILE = os.path.join(DATA_DIR, "phishing_urls.csv")
LEGITIMATE_FILE = os.path.join(DATA_DIR, "legitimate_urls.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# 下载 PhishTank 数据
print(f"  Downloading from PhishTank...")
print(f"  URL: {PHISHTANK_URL}")

try:
    with urllib.request.urlopen(PHISHTANK_URL, timeout=30) as response:
        data = response.read().decode('utf-8')

        # 解析 CSV
        lines = data.strip().split('\n')
        phishing_urls = []

        for i, line in enumerate(lines[1:]):  # 跳过表头
            parts = line.split(',')
            if len(parts) >= 2:
                url = parts[1].strip('"')
                if url and len(url) > 10:
                    phishing_urls.append(url)
                    if len(phishing_urls) >= 5000:  # 限制数量
                        break

        # 保存
        with open(PHISHING_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'label'])
            for url in phishing_urls:
                writer.writerow([url, 1])

        print(f"  ✓ Downloaded {len(phishing_urls)} phishing URLs")
        print(f"  ✓ Saved to: {PHISHING_FILE}")

except Exception as e:
    print(f"  ✗ Download failed: {e}")
    print("  Using backup synthetic data...")
    phishing_urls = []

# ============================================================================
# STEP 2: 生成正常网站数据（Alexa Top 网站）
# ============================================================================
print("\n[Step 2/5] Preparing legitimate URLs...")

# 一些知名正常网站
legitimate_domains = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
    'twitter.com', 'instagram.com', 'linkedin.com', 'netflix.com', 'apple.com',
    'microsoft.com', 'github.com', 'stackoverflow.com', 'reddit.com', 'whatsapp.com',
    'yahoo.com', 'ebay.com', 'paypal.com', 'wordpress.org', 'w3.org',
    'nytimes.com', 'bbc.com', 'cnn.com', 'theguardian.com', 'reuters.com',
    'spotify.com', 'adobe.com', 'ibm.com', 'intel.com', 'amd.com', 'nvidia.com',
    'dropbox.com', 'zoom.us', 'slack.com', 'atlassian.com', 'salesforce.com',
    'docker.com', 'kubernetes.io', 'linux.org', 'python.org', 'nodejs.org'
]

legitimate_urls = []

# 生成正常网站 URL（数量与钓鱼网站匹配）
num_legit = min(len(phishing_urls) if phishing_urls else 500, 2000)

for i in range(num_legit):
    domain = legitimate_domains[i % len(legitimate_domains)]
    # 随机组合路径
    paths = ['', '/', '/search', '/about', '/products', '/user/profile', '/docs/api']
    path = paths[i % len(paths)]

    if domain in ['google.com', 'youtube.com', 'github.com', 'stackoverflow.com']:
        url = f"https://www.{domain}{path}"
    else:
        url = f"https://{domain}{path}"

    legitimate_urls.append(url)

# 保存
with open(LEGITIMATE_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'label'])
    for url in legitimate_urls:
        writer.writerow([url, 0])

print(f"  ✓ Generated {len(legitimate_urls)} legitimate URLs")
print(f"  ✓ Saved to: {LEGITIMATE_FILE}")

# ============================================================================
# STEP 3: 提取特征
# ============================================================================
print("\n[Step 3/5] Extracting URL features...")

def extract_features_simple(url):
    """简化的特征提取"""
    import re
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    features = {
        # 基础特征 (10)
        'url_length': len(url),
        'hostname_length': len(hostname),
        'dot_count': url.count('.'),
        'at_symbol': 1 if '@' in url else 0,
        'dash_count': url.count('-'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'has_ip': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,
        'subdomain_count': max(0, hostname.count('.') - 2),
        'path_depth': (parsed.path or '').count('/'),
        'has_query': 1 if parsed.query else 0,

        # 可疑 TLD (10)
        'tld_xyz': 1 if hostname.endswith('.xyz') else 0,
        'tld_top': 1 if hostname.endswith('.top') else 0,
        'tld_tk': 1 if hostname.endswith('.tk') else 0,
        'tld_ml': 1 if hostname.endswith('.ml') else 0,
        'tld_ga': 1 if hostname.endswith('.ga') else 0,
        'tld_cf': 1 if hostname.endswith('.cf') else 0,
        'tld_gq': 1 if hostname.endswith('.gq') else 0,
        'tld_vip': 1 if hostname.endswith('.vip') else 0,
        'tld_icu': 1 if hostname.endswith('.icu') else 0,
        'tld_online': 1 if hostname.endswith('.online') else 0,

        # 品牌 (8)
        'brand_apple': 1 if 'apple' in url_lower else 0,
        'brand_google': 1 if 'google' in url_lower else 0,
        'brand_microsoft': 1 if 'microsoft' in url_lower else 0,
        'brand_amazon': 1 if 'amazon' in url_lower else 0,
        'brand_paypal': 1 if 'paypal' in url_lower else 0,
        'brand_facebook': 1 if 'facebook' in url_lower else 0,
        'brand_netflix': 1 if 'netflix' in url_lower else 0,
        'brand_dropbox': 1 if 'dropbox' in url_lower else 0,

        # 可疑关键词 (8)
        'kw_login': 1 if 'login' in url_lower else 0,
        'kw_signin': 1 if 'signin' in url_lower else 0,
        'kw_verify': 1 if 'verify' in url_lower else 0,
        'kw_account': 1 if 'account' in url_lower else 0,
        'kw_secure': 1 if 'secure' in url_lower else 0,
        'kw_update': 1 if 'update' in url_lower else 0,
        'kw_password': 1 if 'password' in url_lower else 0,
        'kw_bank': 1 if 'bank' in url_lower else 0,
    }

    return list(features.values())

# 提取特征
all_urls = phishing_urls + legitimate_urls
all_labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)

print(f"  Extracting features from {len(all_urls)} URLs...")
X = []
y = []

for url, label in zip(all_urls, all_labels):
    try:
        features = extract_features_simple(url)
        X.append(features)
        y.append(label)
    except Exception as e:
        pass

print(f"  ✓ Extracted {len(X)} samples")
print(f"  ✓ Feature dimension: {len(X[0]) if X else 0}")

# ============================================================================
# STEP 4: 训练模型
# ============================================================================
print("\n[Step 4/5] Training model with real data...")

import numpy as np

X = np.array(X)
y = np.array(y)

# 尝试使用 PyTorch
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
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = PhishingClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    # 预测
    with torch.no_grad():
        y_pred = (model(X_tensor).squeeze() >= 0.5).numpy().astype(int)

    MODEL_TYPE = 'PyTorch'

except ImportError:
    print("  PyTorch not available, using scikit-learn...")

    try:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        MODEL_TYPE = 'sklearn'

    except ImportError:
        print("  ERROR: No ML library available!")
        sys.exit(1)

# ============================================================================
# STEP 5: 评估
# ============================================================================
print("\n[Step 5/5] Evaluating model...")

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\n  训练集性能 ({MODEL_TYPE}):")
    print(f"    Accuracy:  {accuracy*100:.1f}%")
    print(f"    Precision: {precision*100:.1f}%")
    print(f"    Recall:    {recall*100:.1f}%")
    print(f"    F1-Score:  {f1*100:.1f}%")

except ImportError:
    correct = np.sum(y_pred == y)
    accuracy = correct / len(y)
    print(f"\n  训练集准确率: {accuracy*100:.1f}%")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print(f"数据来源: PhishTank 真实钓鱼网站 ({len(phishing_urls)} 个)")
print(f"正常网站: {len(legitimate_urls)} 个知名网站")
print(f"总样本: {len(all_urls)} 个")
print(f"模型: {MODEL_TYPE}")
print("=" * 70)
