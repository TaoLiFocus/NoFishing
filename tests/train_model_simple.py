# -*- coding: utf-8 -*-
"""
NoFishing Deep Learning Model Training - Simplified Version
简化版深度学习模型训练
"""
import sys
import os

# 设置路径
sys.path.insert(0, 'C:/Users/TaoLi/NoFishing/nofishing-ml-api')

import numpy as np
from datetime import datetime

print("=" * 70)
print(" NoFishing - Deep Learning Model Training & Testing ".center(70))
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: 导入模块
# ============================================================================
print("[Step 1/7] Importing modules...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    print(f"  OK - PyTorch {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
except ImportError:
    HAS_TORCH = False
    print("  WARNING - PyTorch not available, using numpy/sklearn fallback")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
    print(f"  OK - scikit-learn installed")
except ImportError:
    HAS_SKLEARN = False
    print("  WARNING - scikit-learn not available")

# ============================================================================
# STEP 2: 数据集生成
# ============================================================================
print("\n[Step 2/7] Generating phishing dataset...")

def extract_url_features(url):
    """Extract features from URL for ML model"""
    import re
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    features = {}

    # Basic features (0-19)
    features['url_length'] = len(url)
    features['hostname_length'] = len(parsed.hostname or '')
    features['dot_count'] = url.count('.')
    features['at_symbol'] = 1 if '@' in url else 0
    features['dash_count'] = url.count('-')
    features['has_https'] = 1 if parsed.scheme == 'https' else 0
    features['has_ip'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed.hostname or '') else 0
    features['subdomain_count'] = max(0, (parsed.hostname or '').count('.') - 2)
    features['path_depth'] = (parsed.path or '').count('/')
    features['has_query'] = 1 if parsed.query else 0
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = sum(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`"' for c in url)

    # Suspicious TLD (10-19)
    suspicious_tlds = ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq']
    hostname = (parsed.hostname or '').lower()
    for i, tld in enumerate(suspicious_tlds):
        features[f'tld_{i}'] = 1 if hostname.endswith(tld) else 0

    # Brand names (20-27)
    brands = ['apple', 'google', 'microsoft', 'amazon', 'paypal', 'facebook', 'netflix', 'dropbox']
    url_lower = url.lower()
    for i, brand in enumerate(brands):
        features[f'brand_{i}'] = 1 if brand in url_lower else 0

    # Suspicious keywords (28-37)
    keywords = ['login', 'signin', 'verify', 'account', 'secure', 'update', 'password', 'bank', 'crypto']
    for i, kw in enumerate(keywords):
        features[f'keyword_{i}'] = 1 if kw in url_lower else 0

    return list(features.values())

# Generate training data
np.random.seed(42)

# Phishing URL patterns
phishing_patterns = [
    lambda: f"http://{np.random.randint(1,256)}.{np.random.randint(1,256)}.{np.random.randint(1,256)}.{np.random.randint(1,256)}/login",
    lambda: f"http://apple-verify-account.{np.random.choice(['tk', 'ml', 'ga', 'xyz'])}/signin",
    lambda: f"http://@{np.random.choice(['example', 'service', 'company'])}.{np.random.choice(['vip', 'top', 'icu'])}/account",
    lambda: f"http://{np.random.choice(['apple', 'google', 'paypal'])}-{np.random.choice(['secure', 'verify', 'login'])}.ml/confirm",
]

# Legitimate URL patterns
legit_patterns = [
    lambda: f"https://www.{np.random.choice(['google', 'github', 'stackoverflow', 'wikipedia'])}.com/",
    lambda: f"https://api.{np.random.choice(['github', 'stripe', 'aws'])}.com/v1/endpoint",
    lambda: f"https://www.{np.random.choice(['nytimes', 'bbc', 'cnn'])}.com/news/article",
    lambda: f"https://{np.random.choice(['mobile', 'mail', 'www'])}.{np.random.choice(['google', 'microsoft', 'apple'])}.com/",
]

# Generate samples
NUM_SAMPLES = 500
urls = []
labels = []

print(f"  Generating {NUM_SAMPLES} training samples...")

# Phishing samples
for _ in range(NUM_SAMPLES // 2):
    pattern = np.random.choice(phishing_patterns)
    urls.append(pattern())
    labels.append(1)

# Legitimate samples
for _ in range(NUM_SAMPLES // 2):
    pattern = np.random.choice(legit_patterns)
    urls.append(pattern())
    labels.append(0)

# Shuffle
combined = list(zip(urls, labels))
np.random.shuffle(combined)
urls, labels = zip(*combined)

# Extract features
print(f"  Extracting features from {len(urls)} URLs...")
X = np.array([extract_url_features(url) for url in urls])
y = np.array(labels)

print(f"  Feature shape: {X.shape}")
print(f"  Phishing samples: {np.sum(y == 1)}")
print(f"  Legitimate samples: {np.sum(y == 0)}")

# ============================================================================
# STEP 3: 训练模型
# ============================================================================
print("\n[Step 3/7] Training model...")

if HAS_TORCH:
    print("  Using PyTorch Neural Network...")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Define simple model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print(f"    Training for 50 epochs...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

    # Predictions
    with torch.no_grad():
        y_pred_proba = model(X_tensor).squeeze().numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"  Training complete!")

elif HAS_SKLEARN:
    print("  Using Random Forest (sklearn fallback)...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    print(f"  Training complete!")

else:
    print("  ERROR: No ML library available!")
    sys.exit(1)

# ============================================================================
# STEP 4: 评估模型
# ============================================================================
print("\n[Step 4/7] Evaluating model...")

if HAS_SKLEARN:
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1-Score:  {f1*100:.1f}%")

# ============================================================================
# STEP 5: 测试推理
# ============================================================================
print("\n[Step 5/7] Testing inference...")

test_cases = [
    ("https://www.google.com/search", False, "Legitimate search"),
    ("https://github.com/user/repo", False, "Legitimate repo"),
    ("http://apple-verify-account.tk/login", True, "Apple phishing"),
    ("http://192.168.1.1@fake-paypal.ml/signin", True, "PayPal phishing"),
    ("http://@example.com.vip/account", True, "Suspicious structure"),
    ("https://www.netflix.com/watch", False, "Legitimate Netflix"),
]

print(f"\n  {'URL':<50} {'Expected':>10} {'Actual':>10} {'Confidence':>10}")
print("  " + "-" * 85)

for url, expected, desc in test_cases:
    features = np.array([extract_url_features(url)])

    if HAS_TORCH:
        with torch.no_grad():
            prob = model(torch.FloatTensor(features)).squeeze().item()
    else:
        prob = model.predict_proba(features)[0, 1]

    is_phishing = prob >= 0.5
    status = "PHISHING" if is_phishing else "SAFE"
    confidence = f"{prob*100:.0f}%"
    match = "OK" if is_phishing == expected else "WRONG"

    print(f"  {url:<50} {match:>10} {status:>10} {confidence:>10}")

# ============================================================================
# STEP 6: 风险等级测试
# ============================================================================
print("\n[Step 6/7] Risk level distribution...")

if HAS_TORCH:
    with torch.no_grad():
        all_probs = model(X_tensor).squeeze().numpy()
elif HAS_SKLEARN:
    all_probs = model.predict_proba(X)[:, 1]

risk_levels = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

for prob in all_probs:
    if prob >= 0.8:
        risk_levels['CRITICAL'] += 1
    elif prob >= 0.6:
        risk_levels['HIGH'] += 1
    elif prob >= 0.3:
        risk_levels['MEDIUM'] += 1
    else:
        risk_levels['LOW'] += 1

print(f"  Risk Level Distribution:")
for level, count in risk_levels.items():
    pct = count / len(all_probs) * 100
    bar = "█" * int(pct / 2)
    print(f"    {level:8} {pct:5.1f}% {bar}")

# ============================================================================
# STEP 7: 完成
# ============================================================================
print("\n[Step 7/7] Training complete!")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
