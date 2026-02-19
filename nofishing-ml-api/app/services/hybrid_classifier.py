# -*- coding: utf-8 -*-
"""
Hybrid Phishing Classifier
混合式钓鱼检测：本地词法分析 + 远程大模型深度分析
"""
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import requests

from app.config import (
    RISK_THRESHOLDS, PHISHING_THRESHOLD
)

logger = logging.getLogger(__name__)


class LocalURLClassifier:
    """
    本地URL词法特征分类器（快速，毫秒级）

    特征：
    - URL结构特征（20维）
    - 纯本地计算，无需外部API
    - 适合检测明显钓鱼模式
    """

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义模型架构
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

        self.model = URLClassifier(input_dim=20)

        # 加载训练好的权重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded local model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using random weights")

        self.model.eval()

    def extract_features(self, url: str) -> np.ndarray:
        """提取URL词法特征"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
        except:
            from urllib.parse import urlparse
            parsed = urlparse('http://invalid')

        url_lower = url.lower()
        hostname = (parsed.hostname or '').lower()

        return np.array([
            len(url),                                    # 1. URL长度
            len(hostname),                                # 2. 域名长度
            url.count('.'),                               # 3. 点号数量
            1 if '@' in url else 0,                     # 4. @符号
            url.count('-'),                               # 5. 连字符
            url.count('_'),                               # 6. 下划线
            1 if hostname and hostname[0].isdigit() and
               hostname.replace('.', '').isdigit() else 0,  # 7. IP地址
            max(0, hostname.count('.') - 2),              # 8. 子域名层级
            1 if hostname.count('-') > 0 else 0,         # 9. 域名连字符
            1 if hostname.count('_') > 0 else 0,         # 10. 域名下划线
            1 if hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) else 0,  # 11. 免费TLD
            1 if hostname.endswith('.xyz') else 0,        # 12. .xyz TLD
            1 if any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']) else 0,  # 13. 常见TLD
            1 if parsed.scheme == 'https' else 0,         # 14. HTTPS
            1 if parsed.port and parsed.port not in [80, 443] else 0,  # 15. 非标准端口
            len(parsed.path) if parsed.path else 0,        # 16. 路径长度
            (parsed.path or '').count('/'),                # 17. 路径深度
            1 if parsed.query else 0,                     # 18. 查询参数
            1 if any(w in url_lower for w in ['login', 'signin', 'account', 'verify', 'secure']) else 0,  # 19. 敏感词
            1 if any(w in url_lower for w in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,  # 20. 品牌词
        ], dtype=np.float32)

    def predict(self, url: str) -> Dict[str, Any]:
        """快速本地预测"""
        import time
        start = time.time()

        # 提取特征
        features = self.extract_features(url)

        # 模型推理
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prob = self.model(features_tensor).squeeze().item()

        elapsed = int((time.time() - start) * 1000)

        return {
            'probability': float(prob),
            'is_phishing': prob >= PHISHING_THRESHOLD,
            'confidence': prob,
            'risk_level': self._get_risk_level(prob),
            'processing_time_ms': elapsed,
            'method': 'local_lexical'
        }

    def _get_risk_level(self, confidence: float) -> str:
        if confidence >= RISK_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        elif confidence >= RISK_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif confidence >= RISK_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'


class RemoteLLMAnalyzer:
    """
    远程大模型分析器（慢但准确）

    使用场景：
    - 本地模型判断为"中等风险"时
    - 需要更深入分析页面内容时

    支持的模型：
    - OpenAI API (GPT-4)
    - 可扩展其他模型
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = 'openai'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.provider = provider

        if not self.api_key:
            logger.warning("No API key provided for remote LLM")

        # API配置
        self.api_endpoint = {
            'openai': 'https://api.openai.com/v1/chat/completions',
        }.get(provider)

        self.model_name = {
            'openai': 'gpt-4o-mini',  # 性价比高
        }.get(provider)

    def analyze(self, url: str, fetch_content: bool = False) -> Dict[str, Any]:
        """
        深度分析URL和可选的页面内容

        Args:
            url: 要分析的URL
            fetch_content: 是否抓取页面内容（可选，需要额外时间）

        Returns:
            {
                'probability': float,
                'is_phishing': bool,
                'reasoning': str,
                'risk_indicators': list,
                'processing_time_ms': int,
                'method': 'remote_llm'
            }
        """
        import time
        start = time.time()

        if not self.api_key:
            return {
                'probability': 0.5,
                'is_phishing': False,
                'reasoning': 'LLM API key not configured',
                'risk_indicators': [],
                'processing_time_ms': 0,
                'method': 'remote_llm',
                'error': 'no_api_key'
            }

        # 构建分析提示
        prompt = self._build_analysis_prompt(url)

        try:
            # 调用LLM API
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'model': self.model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a cybersecurity expert specializing in phishing detection. '
                                   'Analyze URLs and identify potential phishing indicators.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.1,  # 低温度以获得更确定的结果
                'max_tokens': 500
            }

            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=10  # 10秒超时
            )

            if response.status_code == 200:
                result = response.json()

                # 解析响应
                content = result['choices'][0]['message']['content']

                # 提取结构化结果
                return self._parse_llm_response(content, time.time() - start)

            else:
                logger.error(f"LLM API error: {response.status_code}")
                return self._fallback_response(url, time.time() - start, f"API error: {response.status_code}")

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_response(url, time.time() - start, str(e))

    def _build_analysis_prompt(self, url: str) -> str:
        """构建LLM分析提示"""
        return f"""Analyze this URL for phishing indicators: {url}

Check for:
1. **Suspicious patterns**:
   - Typosquatting (e.g., goggle.com instead of google.com)
   - Brand impersonation (e.g., apple-security.tk)
   - Free TLD usage (.tk, .ml, .ga, .cf, .gq, .xyz)
   - IP address instead of domain
   - Excessive subdomains or hyphens
   - Suspicious keywords (login, signin, verify, secure, account)

2. **Legitimate indicators**:
   - Proper HTTPS with valid certificate
   - Established domain (.com, .org) from known brands
   - Clean URL structure
   - No deceptive patterns

3. **Provide your assessment in this exact format**:
   RISK: [HIGH/MEDIUM/LOW/SAFE]
   CONFIDENCE: [0-100]
   REASONING: [brief explanation]
   INDICATORS: [comma-separated list of findings]

Example response:
   RISK: HIGH
   CONFIDENCE: 85
   REASONING: This URL uses a free TLD (.tk) with brand impersonation (apple-verify)
   INDICATORS: free TLD, brand impersonation, suspicious keyword "verify"
"""

    def _parse_llm_response(self, content: str, elapsed: float) -> Dict[str, Any]:
        """解析LLM响应"""
        import re

        # 默认值
        risk = 'UNKNOWN'
        confidence = 50
        reasoning = content
        indicators = []

        # 提取RISK
        risk_match = re.search(r'RISK:\s*(HIGH|MEDIUM|LOW|SAFE)', content, re.IGNORECASE)
        if risk_match:
            risk = risk_match.group(1).upper()

        # 提取CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', content)
        if conf_match:
            confidence = int(conf_match.group(1))
            confidence = min(100, max(0, confidence))

        # 提取REASONING
        reason_match = re.search(r'REASONING:\s*(.+?)(?=INDICATORS:|$)', content, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        else:
            reasoning = content

        # 提取INDICATORS
        indicators_match = re.search(r'INDICATORS:\s*(.+)', content)
        if indicators_match:
            indicators_str = indicators_match.group(1).strip()
            indicators = [i.strip() for i in indicators_str.split(',') if i.strip()]

        # 转换为概率
        risk_to_prob = {
            'HIGH': 0.85,
            'MEDIUM': 0.60,
            'LOW': 0.30,
            'SAFE': 0.10,
            'UNKNOWN': 0.50
        }
        probability = risk_to_prob.get(risk, 0.50)

        return {
            'probability': probability,
            'is_phishing': probability >= PHISHING_THRESHOLD,
            'confidence': confidence / 100.0,
            'risk_level': risk,
            'reasoning': reasoning,
            'risk_indicators': indicators,
            'processing_time_ms': int(elapsed * 1000),
            'method': 'remote_llm'
        }

    def _fallback_response(self, url: str, elapsed: float, error: str) -> Dict[str, Any]:
        """LLM调用失败时的回退响应"""
        return {
            'probability': 0.5,
            'is_phishing': False,
            'confidence': 0.5,
            'risk_level': 'UNKNOWN',
            'reasoning': f'Analysis unavailable: {error}',
            'risk_indicators': [],
            'processing_time_ms': int(elapsed * 1000),
            'method': 'remote_llm',
            'error': error
        }


class HybridPhishingClassifier:
    """
    混合钓鱼检测分类器

    策略：
    1. 本地快速分析（所有URL）
    2. 如果本地置信度 < 0.7 且 > 0.3，调用远程LLM深度分析
    3. 综合两个结果给出最终判断
    """

    def __init__(self,
                 local_model_path: str,
                 llm_api_key: Optional[str] = None,
                 llm_provider: str = 'openai'):

        self.local_classifier = LocalURLClassifier(local_model_path)
        self.remote_analyzer = RemoteLLMAnalyzer(api_key=llm_api_key, provider=llm_provider)

        # 混合策略配置
        self.local_confidence_threshold = 0.7  # 本地置信度高于此值则不需要远程分析
        self.remote_trigger_threshold = 0.3   # 本地置信度高于此值则不触发远程分析（明显安全）

        logger.info("Hybrid classifier initialized")

    def classify(self, url: str, enable_remote: bool = True) -> Dict[str, Any]:
        """
        混合分类

        Args:
            url: 要检测的URL
            enable_remote: 是否启用远程LLM分析

        Returns:
            {
                'is_phishing': bool,
                'probability': float,
                'confidence': float,
                'risk_level': str,
                'method_used': str,
                'local_result': dict,
                'remote_result': dict or None,
                'processing_time_ms': int,
                'final_reasoning': str
            }
        """
        import time
        start_time = time.time()

        # 第一阶段：本地快速分析
        local_result = self.local_classifier.predict(url)
        local_prob = local_result['probability']

        logger.info(f"Local analysis: {local_prob:.3f} ({local_result['risk_level']}) in {local_result['processing_time_ms']}ms")

        # 决策逻辑
        remote_result = None
        final_probability = local_prob
        final_reasoning = f"Local analysis determined risk level: {local_result['risk_level']}"
        method_used = "local_only"

        # 情况1：本地高置信度钓鱼 -> 直接返回
        if local_prob >= 0.8:
            final_reasoning = f"High confidence phishing detected by local analysis: {local_result['risk_level']}"

        # 情况2：本地高置信度安全 -> 直接返回
        elif local_prob <= self.remote_trigger_threshold:
            final_reasoning = f"High confidence safe URL by local analysis: {local_result['risk_level']}"

        # 情况3：中等置信度 + 启用远程 -> 调用LLM深度分析
        elif enable_remote and local_prob > self.remote_trigger_threshold and local_prob < self.local_confidence_threshold:
            logger.info(f"Triggering remote LLM analysis for uncertain case: {local_prob:.3f}")
            method_used = "hybrid_local_llm"

            remote_result = self.remote_analyzer.analyze(url)
            remote_prob = remote_result['probability']

            logger.info(f"Remote analysis: {remote_prob:.3f} ({remote_result.get('risk_level', 'UNKNOWN')}) "
                       f"in {remote_result['processing_time_ms']}ms")

            # 综合两个结果（加权平均，远程权重更高）
            final_probability = (local_prob * 0.3) + (remote_prob * 0.7)

            reasoning_parts = [
                f"Local: {local_prob:.0%} ({local_result['risk_level']})",
            ]
            if 'error' not in remote_result:
                reasoning_parts.extend([
                    f"Remote: {remote_prob:.0%} ({remote_result.get('risk_level', 'UNKNOWN')})",
                    f"Reasoning: {remote_result.get('reasoning', 'N/A')[:100]}"
                ])
            else:
                reasoning_parts.append(f"Remote analysis failed: {remote_result.get('error', 'Unknown error')}")

            final_reasoning = " | ".join(reasoning_parts)

        # 情况4：远程未启用 -> 使用本地结果
        else:
            final_reasoning = f"Local analysis (remote not enabled or not triggered)"

        # 确定最终分类
        is_phishing = final_probability >= PHISHING_THRESHOLD
        risk_level = self._get_risk_level(final_probability)

        total_time = int((time.time() - start_time) * 1000)

        result = {
            'is_phishing': is_phishing,
            'probability': float(final_probability),
            'confidence': abs(final_probability - 0.5) * 2,  # 转换为0-1置信度
            'risk_level': risk_level,
            'method_used': method_used,
            'local_result': local_result,
            'remote_result': remote_result,
            'processing_time_ms': total_time,
            'final_reasoning': final_reasoning,
            'url': url,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Final classification: {is_phishing} (prob: {final_probability:.3f}, "
                   f"risk: {risk_level}, time: {total_time}ms)")

        return result

    def _get_risk_level(self, probability: float) -> str:
        if probability >= 0.8:
            return 'CRITICAL'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'SAFE'


# ============================================================================
# Singleton
# ============================================================================
_classifier = None


def get_hybrid_classifier(local_model_path: str = None,
                        llm_api_key: str = None,
                        llm_provider: str = 'openai') -> HybridPhishingClassifier:
    """获取或创建混合分类器实例"""
    global _classifier

    if _classifier is None:
        if local_model_path is None:
            from app.config import MODEL_PATH
            local_model_path = str(MODEL_PATH)

        _classifier = HybridPhishingClassifier(
            local_model_path=local_model_path,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )

    return _classifier
