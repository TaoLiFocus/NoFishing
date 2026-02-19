# -*- coding: utf-8 -*-
"""
Compare Feature Extraction
对比特征提取方法
"""
import torch
import numpy as np
import re
from urllib.parse import urlparse
import sys
sys.path.insert(0, '.')

from models.phishing_classifier import PhishingClassifierModel

wrapper = PhishingClassifierModel()

url = "http://apple-verify.tk/account"

print("=" * 70)
print(f"Comparing Feature Extraction for: {url}")
print("=" * 70)

# ============================================================================
# Method 1: Direct feature extraction (works)
# ============================================================================
print("\n[Method 1] Direct extraction (working)")

def extract_direct(url):
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse('http://invalid')

    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    return np.array([
        len(url),                                    # 1
        len(hostname),                                # 2
        url.count('.'),                               # 3
        1 if '@' in url else 0,                     # 4
        url.count('-'),                               # 5
        url.count('_'),                               # 6
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,  # 7
        max(0, hostname.count('.') - 2),              # 8
        1 if hostname.count('-') > 0 else 0,         # 9
        1 if hostname.count('_') > 0 else 0,         # 10
        1 if hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) else 0,  # 11
        1 if hostname.endswith('.xyz') else 0,        # 12
        1 if any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']) else 0,  # 13
        1 if parsed.scheme == 'https' else 0,         # 14
        1 if parsed.port and parsed.port not in [80, 443] else 0,  # 15
        len(parsed.path) if parsed.path else 0,        # 16
        (parsed.path or '').count('/'),                # 17
        1 if parsed.query else 0,                     # 18
        1 if any(word in url_lower for word in ['login', 'signin', 'account', 'verify', 'secure']) else 0,  # 19
        1 if any(word in url_lower for word in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,  # 20
    ], dtype=np.float32)

features_direct = extract_direct(url)
print(f"  Features: {features_direct}")

# ============================================================================
# Method 2: Using wrapper's _prepare_feature_vector
# ============================================================================
print("\n[Method 2] Wrapper _prepare_feature_vector")

features_wrapper_dict = wrapper.processor.extract_features(url)
features_wrapper_tensor = wrapper._prepare_feature_vector({'url': url})
features_wrapper = features_wrapper_tensor.cpu().numpy().flatten()

print(f"  Features dict: {features_wrapper_dict}")
print(f"  Features tensor: {features_wrapper}")

# ============================================================================
# Method 3: Direct prediction
# ============================================================================
print("\n[Method 3] Direct prediction")

from models.phishing_classifier import PhishingClassifier
import torch.nn as nn

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

from app.config import MODEL_PATH
model = URLClassifier(input_dim=20)
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    prob_direct = model(torch.FloatTensor(features_direct).unsqueeze(0)).item()
    prob_wrapper = model(features_wrapper_tensor).item()

print(f"  Direct features -> prob: {prob_direct:.4f}")
print(f"  Wrapper features -> prob: {prob_wrapper:.4f}")

# ============================================================================
# Feature comparison
# ============================================================================
print("\n[Feature Comparison]")
print(f"  {'Index':<10} {'Direct':<15} {'Wrapper':<15} {'Diff':<15}")
print("  " + "-" * 60)
for i in range(20):
    diff = features_direct[i] - features_wrapper[i]
    mark = " ***" if abs(diff) > 0.1 else ""
    print(f"  {i:<10} {features_direct[i]:<15.6f} {features_wrapper[i]:<15.6f} {diff:<15.6f}{mark}")

print("\n" + "=" * 70)
