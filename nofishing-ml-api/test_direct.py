# -*- coding: utf-8 -*-
"""
Direct Model Test
直接测试模型加载和推理
"""
import torch
import torch.nn as nn
import numpy as np
import re
from urllib.parse import urlparse

print("=" * 70)
print("Direct Model Test")
print("=" * 70)

# ============================================================================
# Define the exact model architecture from training
# ============================================================================
class URLClassifier(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# Load the trained model
# ============================================================================
print("\n[Loading Model]")

MODEL_PATH = "models/phishing_classifier_url_only.pt"
model = URLClassifier(input_dim=20)
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)
model.eval()

print(f"  Model loaded from: {MODEL_PATH}")
print(f"  Model type: {type(model).__name__}")

# ============================================================================
# Feature extraction (must match training)
# ============================================================================
def extract_url_features(url):
    """Extract 20 URL-lexical features (must match training)"""
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    return [
        len(url),                                    # 1. URL长度
        len(hostname),                                # 2. 域名长度
        url.count('.'),                               # 3. 点号数量
        1 if '@' in url else 0,                     # 4. 是否有@符号
        url.count('-'),                               # 5. 连字符数量
        url.count('_'),                               # 6. 下划线数量
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,  # 7. 是否为IP地址
        max(0, hostname.count('.') - 2),              # 8. 子域名层级
        1 if hostname.count('-') > 0 else 0,         # 9. 域名是否有连字符
        1 if hostname.count('_') > 0 else 0,         # 10. 域名是否有下划线
        1 if hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) else 0,  # 11. 免费TLD
        1 if hostname.endswith('.xyz') else 0,        # 12. .xyz TLD
        1 if any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']) else 0,  # 13. 常见TLD
        1 if parsed.scheme == 'https' else 0,         # 14. 是否HTTPS
        1 if parsed.port and parsed.port not in [80, 443] else 0,  # 15. 非标准端口
        len(parsed.path) if parsed.path else 0,        # 16. 路径长度
        (parsed.path or '').count('/'),                # 17. 路径深度
        1 if parsed.query else 0,                     # 18. 是否有查询参数
        1 if any(word in url_lower for word in ['login', 'signin', 'account', 'verify', 'secure']) else 0,  # 19. 敏感词
        1 if any(word in url_lower for word in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,  # 20. 品牌词
    ]

# ============================================================================
# Test URLs
# ============================================================================
print("\n[Classification Test]")

test_urls = [
    ("https://www.google.com/search", 0, "Legitimate Google"),
    ("http://192.168.1.1/login", 1, "IP Address Login"),
    ("http://apple-verify.tk/account", 1, "Apple Phishing"),
    ("https://github.com/user/repo", 0, "Legitimate GitHub"),
    ("http://paypal-secure.ml/signin", 1, "PayPal Phishing"),
    ("https://www.amazon.com/dp/product", 0, "Legitimate Amazon"),
    ("http://google-secure.cf/login", 1, "Google Phishing"),
    ("https://stackoverflow.com/questions", 0, "Legitimate StackOverflow"),
]

print(f"\n  {'URL':<50} {'Expected':>10} {'Actual':>10} {'Conf':>10}")
print("  " + "-" * 85)

correct = 0
total = len(test_urls)

for url, expected, desc in test_urls:
    features = np.array([extract_url_features(url)], dtype=np.float32)
    features_tensor = torch.FloatTensor(features)

    with torch.no_grad():
        prob = model(features_tensor).squeeze().item()

    is_phishing = prob >= 0.5
    actual = 1 if is_phishing else 0
    match = "OK" if actual == expected else "WRONG"
    status = "PHISH" if is_phishing else "SAFE"
    conf = f"{prob*100:.0f}%"

    if actual == expected:
        correct += 1

    print(f"  {url:<50} {match:>10} {status:>10} {conf:>10} - {desc}")

accuracy = (correct / total) * 100
print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{total})")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print(f"Status: {'PASS' if accuracy >= 90 else 'FAIL'}")
print("=" * 70)
