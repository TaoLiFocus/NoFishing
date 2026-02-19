# -*- coding: utf-8 -*-
"""
Simple URL-Only Phishing Classifier
纯URL词法特征分类器（职责单一）
"""
import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from app.config import PHISHING_THRESHOLD

logger = logging.getLogger(__name__)


class URLPhishingClassifier:
    """
    URL词法特征分类器

    特点：
    - 仅使用URL本身，不获取网页内容
    - 快速推理（毫秒级）
    - 20维词法特征
    """

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型架构（与训练时完全一致）
        class Model(nn.Module):
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

        self.model = Model(input_dim=20)

        # 加载训练好的权重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using random weights")

        self.model.eval()

    def extract_features(self, url: str) -> np.ndarray:
        """
        提取URL词法特征（20维）

        特征列表：
        1. URL长度
        2. 域名长度
        3. 点号数量
        4. @符号存在
        5. 连字符数量
        6. 下划线数量
        7. 是否为IP地址
        8. 子域名层级
        9. 域名连字符存在
        10. 域名下划线存在
        11. 免费TLD (.tk, .ml, .ga, .cf, .gq)
        12. .xyz TLD
        13. 常见TLD (.com, .org, .net, .edu)
        14. HTTPS协议
        15. 非标准端口
        16. 路径长度
        17. 路径深度
        18. 查询参数存在
        19. 敏感关键词
        20. 品牌关键词
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
        except:
            from urllib.parse import urlparse
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
            1 if hostname and hostname.replace('.', '').isdigit() else 0,  # 7 IP
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
            1 if any(w in url_lower for w in ['login', 'signin', 'account', 'verify', 'secure']) else 0,  # 19
            1 if any(w in url_lower for w in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,  # 20
        ], dtype=np.float32)

    def classify(self, url: str) -> Dict[str, Any]:
        """
        对URL进行分类

        Args:
            url: 要检测的URL

        Returns:
            {
                'url': str,
                'is_phishing': bool,
                'probability': float (0-1),
                'risk_level': str,
                'features': dict,
                'processing_time_ms': int
            }
        """
        import time
        start = time.time()

        # 提取特征
        features = self.extract_features(url)

        # 模型推理
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            probability = self.model(features_tensor).squeeze().item()

        # 确定分类
        is_phishing = probability >= PHISHING_THRESHOLD

        # 计算风险等级
        if probability >= 0.8:
            risk_level = 'CRITICAL'
        elif probability >= 0.6:
            risk_level = 'HIGH'
        elif probability >= 0.4:
            risk_level = 'MEDIUM'
        elif probability >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'SAFE'

        processing_time = int((time.time() - start) * 1000)

        # 提取特征详情（用于调试和显示）
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()

        feature_details = {
            'url_length': len(url),
            'domain_length': len(hostname),
            'dot_count': url.count('.'),
            'has_at_symbol': '@' in url,
            'dash_count': url.count('-'),
            'underscore_count': url.count('_'),
            'is_ip_address': hostname.replace('.', '').isdigit(),
            'subdomain_level': max(0, hostname.count('.') - 2),
            'has_dash_in_domain': '-' in hostname,
            'has_underscore_in_domain': '_' in hostname,
            'is_free_tld': hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq', '.xyz')),
            'is_common_tld': any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']),
            'is_https': parsed.scheme == 'https',
            'has_non_standard_port': parsed.port not in [None, 80, 443],
            'has_suspicious_keyword': any(w in url.lower() for w in ['login', 'signin', 'account', 'verify', 'secure']),
            'has_brand_keyword': any(w in url.lower() for w in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']),
        }

        return {
            'url': url,
            'is_phishing': is_phishing,
            'probability': float(probability),
            'risk_level': risk_level,
            'features': feature_details,
            'processing_time_ms': processing_time
        }


# ============================================================================
# Singleton
# ============================================================================
_classifier = None


def get_classifier() -> URLPhishingClassifier:
    """获取或创建分类器实例"""
    global _classifier

    if _classifier is None:
        from app.config import MODEL_PATH
        _classifier = URLPhishingClassifier(str(MODEL_PATH))
        logger.info("URL Phishing Classifier initialized")

    return _classifier
