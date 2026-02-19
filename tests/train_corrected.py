# -*- coding: utf-8 -*-
"""
NoFishing - Corrected Training with Real ARFF Dataset
使用真实钓鱼网站数据集训练模型（修正版）
"""
import sys
import os
import re
from urllib.parse import urlparse

print("=" * 70)
print("NoFishing - Real Dataset Training (Corrected)")
print("=" * 70)

# ============================================================================
# 加载 ARFF 数据集
# ============================================================================
print("\n[Step 1/5] Loading ARFF dataset...")

ARFF_FILE = "D:/BrowserDownload/FishingSource/Training Dataset.arff"

def load_arff_corrected(filepath):
    """Load ARFF format file - 30 features + 1 label (Result)"""
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

    print(f"  Total attributes in ARFF: {len(attributes)}")
    print(f"  Features (first 30): {attributes[:30]}")
    print(f"  Label (31st): {attributes[30] if len(attributes) > 30 else 'N/A'}")

    # 解析数据 - 30 features + 1 label
    X = []  # features
    y = []  # labels

    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('%'):
            # ARFF 格式: -1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1
            values = line.split(',')

            # 前30个值作为特征
            features = []
            for v in values[:30]:
                v = v.strip()
                if v == '?':
                    features.append(0)
                else:
                    try:
                        features.append(int(v))
                    except ValueError:
                        features.append(0)

            # 第31个值作为标签 (Result: -1=legitimate, 1=phishing)
            if len(values) > 30:
                label_value = values[30].strip()
                if label_value == '?':
                    label = 0
                else:
                    try:
                        label = int(label_value)
                        # 转换: -1 -> 0 (legitimate), 1 -> 1 (phishing)
                        label = 1 if label == 1 else 0
                    except ValueError:
                        label = 0
            else:
                label = 0

            X.append(features)
            y.append(label)

    return X, y, attributes

X, y, attributes = load_arff_corrected(ARFF_FILE)

print(f"\n  Loaded {len(X)} samples")
print(f"  Feature dimension: {len(X[0]) if X else 0}")
print(f"  Phishing samples: {sum(y)}")
print(f"  Legitimate samples: {len(y) - sum(y)}")

# ============================================================================
# 特征提取（用于新 URL 推理）
# ============================================================================
print("\n[Step 2/5] Setting up feature extraction for inference...")

def extract_features_for_inference(url):
    """
    Extract features matching the 30 ARFF features for inference
    注意：这是一个简化版本，因为ARFF的许多特征需要网页内容分析
    """
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    return [
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else -1,  # 1. having_IP_Address
        1 if len(url) > 75 else (-1 if len(url) < 54 else 0),  # 2. URL_Length (simplified)
        -1,  # 3. Shortining_Service (需要检查短链接服务)
        1 if '@' in url else -1,  # 4. having_At_Symbol
        1 if '//' in url[8:] else -1,  # 5. double_slash_redirecting
        1 if '-' in hostname else -1,  # 6. Prefix_Suffix
        1 if hostname.count('.') > 2 else (0 if hostname.count('.') == 2 else -1),  # 7. having_Sub_Domain
        -1,  # 8. SSLfinal_State (需要SSL证书验证)
        -1,  # 9. Domain_registeration_length (需要WHOIS查询)
        -1,  # 10. Favicon (需要网页内容)
        1 if parsed.port and parsed.port not in [80, 443] else -1,  # 11. port
        -1,  # 12. HTTPS_token
        -1,  # 13. Request_URL (需要网页内容)
        -1,  # 14. URL_of_Anchor (需要网页内容)
        -1,  # 15. Links_in_tags (需要网页内容)
        -1,  # 16. SFH (需要网页内容)
        -1,  # 17. Submitting_to_email (需要网页内容)
        -1,  # 18. Abnormal_URL (需要网页内容)
        -1,  # 19. Redirect (需要HTTP请求)
        1 if 'onmouseover' in url_lower else -1,  # 20. on_mouseover
        -1,  # 21. RightClick (需要网页内容)
        -1,  # 22. popUpWidnow (需要网页内容)
        1 if 'iframe' in url_lower else -1,  # 23. Iframe
        -1,  # 24. age_of_domain (需要WHOIS查询)
        -1,  # 25. DNSRecord (需要DNS查询)
        -1,  # 26. web_traffic (需要Alexa排名)
        -1,  # 27. Page_Rank (需要Google PR)
        -1,  # 28. Google_Index (需要搜索检查)
        -1,  # 29. Links_pointing_to_page (需要反向链接检查)
        -1,  # 30. Statistical_report (需要网页内容)
    ]

print(f"  Feature extraction ready (30 dimensions)")

# ============================================================================
# 准备训练数据
# ============================================================================
print("\n[Step 3/5] Preparing training data...")

import numpy as np

X_arr = np.array(X, dtype=np.float32)
y_arr = np.array(y, dtype=np.float32)

print(f"  Training samples: {len(X_arr)}")
print(f"  Feature dimension: {X_arr.shape[1]}")
print(f"  Label distribution: Phishing={sum(y_arr)}, Legitimate={len(y_arr)-sum(y_arr)}")

# ============================================================================
# 训练 PyTorch 模型
# ============================================================================
print("\n[Step 4/5] Training PyTorch model...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"  Using PyTorch {torch.__version__}")

    # 转换为张量
    X_tensor = torch.FloatTensor(X_arr)
    y_tensor = torch.FloatTensor(y_arr)

    # 定义模型
    class PhishingClassifier(nn.Module):
        def __init__(self, input_dim=30):
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

    model = PhishingClassifier(input_dim=30)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_tensor).squeeze()
        y_pred = (y_pred_prob >= 0.5).float()

    accuracy = (y_pred == y_tensor).float().mean()

    # 计算每个类别的准确率
    phishing_indices = (y_tensor == 1)
    legit_indices = (y_tensor == 0)

    phishing_acc = (y_pred[phishing_indices] == y_tensor[phishing_indices]).float().mean() if phishing_indices.any() else 0
    legit_acc = (y_pred[legit_indices] == y_tensor[legit_indices]).float().mean() if legit_indices.any() else 0

    print(f"\n  Training Accuracy: {accuracy*100:.1f}%")
    print(f"    Phishing Class Accuracy: {phishing_acc*100:.1f}%")
    print(f"    Legitimate Class Accuracy: {legit_acc*100:.1f}%")

    # 保存模型
    MODEL_PATH = "C:/Users/TaoLi/NoFishing/nofishing-ml-api/models/phishing_classifier_trained.pt"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    HAS_TORCH = True

except ImportError:
    print("  PyTorch not available, using sklearn...")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_arr, y_arr)
    y_pred = model.predict(X_arr)

    accuracy = np.mean(y_pred == y_arr)
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
    ("http://paypal-secure.ml/signin", 1, "PayPal Phishing"),
]

print("\nTest Results:")
print(f"  {'URL':<50} {'Expected':>10} {'Actual':>10}")
print("  " + "-" * 75)

for url, expected, desc in test_cases:
    features = np.array([extract_features_for_inference(url)], dtype=np.float32)

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
print(f"  Samples: {len(X)}")
print(f"  Features: 30 (excluded 'Result' label)")
print(f"  Phishing: {sum(y)}")
print(f"  Legitimate: {len(y) - sum(y)}")
print(f"  Model: PyTorch Neural Network" if HAS_TORCH else "sklearn Random Forest")
print("=" * 70)
