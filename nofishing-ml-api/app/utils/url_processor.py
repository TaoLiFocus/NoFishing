"""
URL Feature Processor
Extracts lexical and structural features from URLs
"""
import re
import logging
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

import validators

logger = logging.getLogger(__name__)


class URLProcessor:
    """Extract features from URLs for phishing detection"""

    # Suspicious TLDs
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.win', '.loan', '.club', '.online',
        '.vip', '.work', '.shop', '.site', '.icu',
        '.tk', '.ml', '.ga', '.cf', '.gq', '.pp.ua'
    }

    # Suspicious keywords
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'verify', 'account', 'secure',
        'update', 'confirm', 'bank', 'wallet', 'crypto',
        'password', 'credential', 'auth', 'token', 'free',
        'gift', 'winner', 'prize', 'claim', 'urgent'
    ]

    # Brand names for impersonation detection
    BRAND_NAMES = [
        'apple', 'google', 'microsoft', 'amazon', 'facebook',
        'paypal', 'dropbox', 'netflix', 'spotify', 'instagram',
        'twitter', 'linkedin', 'yahoo', 'ebay', 'steam',
        'bankofamerica', 'chase', 'wellsfargo', 'citibank'
    ]

    # IP address pattern
    IP_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

    # Hex encoding patterns
    HEX_PATTERN = re.compile(r'%[0-9a-fA-F]{2}')

    def __init__(self):
        self.logger = logger

    def extract_features(self, url: str) -> Dict[str, Any]:
        """
        Extract all features from a URL

        Args:
            url: URL string to analyze

        Returns:
            Dictionary of extracted features
        """
        if not url:
            raise ValueError("URL cannot be empty")

        # Normalize URL
        url = url.strip().lower()

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            self.logger.warning(f"Failed to parse URL: {url} - {e}")
            return {'error': 'invalid_url', 'url': url}

        features = {'original_url': url}

        # Basic URL components
        features.update(self._extract_basic_features(parsed))

        # Domain features
        features.update(self._extract_domain_features(parsed))

        # Path features
        features.update(self._extract_path_features(parsed))

        # Query string features
        features.update(self._extract_query_features(parsed))

        # Special patterns
        features.update(self._extract_pattern_features(url, parsed))

        # Calculate heuristic score
        features['heuristic_score'] = self._calculate_heuristic_score(features)

        return features

    def _extract_basic_features(self, parsed) -> Dict[str, Any]:
        """Extract basic URL features"""
        return {
            'scheme': parsed.scheme,
            'hostname': parsed.hostname or '',
            'port': parsed.port or -1,
            'url_length': len(parsed.geturl()),
            'has_netloc': bool(parsed.netloc),
        }

    def _extract_domain_features(self, parsed) -> Dict[str, Any]:
        """Extract domain-related features"""
        hostname = parsed.hostname or ''
        features = {}

        # Domain length
        features['domain_length'] = len(hostname)

        # Subdomain count
        parts = hostname.split('.')
        features['subdomain_count'] = max(0, len(parts) - 2)

        # TLD analysis
        features['tld'] = parts[-1] if len(parts) > 1 else ''
        features['has_suspicious_tld'] = any(
            hostname.endswith(tld) for tld in self.SUSPICIOUS_TLDS
        )

        # IP address check
        features['has_ip_address'] = bool(self.IP_PATTERN.match(hostname))

        # HTTPS check
        features['has_https'] = parsed.scheme == 'https'

        # Dash in domain
        features['has_dash_in_domain'] = '-' in hostname

        # Numeric domain
        features['has_numeric_domain'] = any(c.isdigit() for c in hostname)

        # Subdomain length ratio
        if len(parts) > 2:
            subdomain = '.'.join(parts[:-2])
            features['subdomain_length_ratio'] = len(subdomain) / len(hostname) if hostname else 0
        else:
            features['subdomain_length_ratio'] = 0

        return features

    def _extract_path_features(self, parsed) -> Dict[str, Any]:
        """Extract path-related features"""
        path = parsed.path or ''
        features = {}

        features['path_length'] = len(path)
        features['path_depth'] = path.count('/')
        features['has_path'] = bool(path)

        # Check for suspicious keywords in path
        features['has_suspicious_keyword'] = any(
            kw in path for kw in self.SUSPICIOUS_KEYWORDS
        )

        # File extensions
        if '.' in path:
            features['file_extension'] = path.rsplit('.', 1)[-1]
        else:
            features['file_extension'] = ''

        return features

    def _extract_query_features(self, parsed) -> Dict[str, Any]:
        """Extract query string features"""
        query = parsed.query or ''
        features = {'has_query': bool(query)}

        if query:
            params = parse_qs(query)
            features['query_param_count'] = len(params)

            # Check for token-like parameters
            token_params = {'token', 'key', 'session', 'auth', 'api_key'}
            features['has_token_param'] = any(
                param in token_params for param in params.keys()
            )
        else:
            features['query_param_count'] = 0
            features['has_token_param'] = False

        return features

    def _extract_pattern_features(self, url: str, parsed) -> Dict[str, Any]:
        """Extract special pattern features"""
        features = {}

        # @ symbol (often indicates username:password@host pattern)
        features['has_at_symbol'] = '@' in url

        # Double slash after scheme
        features['has_double_slash'] = '//' in url[8:]  # After scheme

        # Percent encoding
        features['percent_encode_count'] = len(self.HEX_PATTERN.findall(url))
        features['has_percent_encoding'] = features['percent_encode_count'] > 0

        # Dots count
        features['dot_count'] = url.count('.')

        # Digits count
        features['digit_count'] = sum(c.isdigit() for c in url)

        # Special characters count
        special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`"'
        features['special_char_count'] = sum(c in special_chars for c in url)

        # Non-ASCII characters
        features['has_non_ascii'] = any(ord(c) > 127 for c in url)

        # Freenom subdomains (often used by phishers)
        freenom_domains = ['.tk', '.ml', '.ga', '.cf', '.gq']
        hostname = parsed.hostname or ''
        features['has_freenom_domain'] = any(
            hostname.endswith(d) for d in freenom_domains
        )

        # Check for brand impersonation in hostname and full URL
        # Convert to lowercase for matching
        url_lower = url.lower()
        hostname_lower = hostname.lower()
        features['has_brand_name'] = any(
            brand in hostname_lower or brand in url_lower
            for brand in self.BRAND_NAMES
        )

        return features

    def _calculate_heuristic_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate a simple heuristic score (0-1)
        This is NOT the ML prediction, just a rule-based score
        """
        score = 0.0

        # IP address
        if features.get('has_ip_address'):
            score += 0.3

        # Suspicious TLD
        if features.get('has_suspicious_tld'):
            score += 0.2

        # No HTTPS
        if not features.get('has_https'):
            score += 0.1

        # Suspicious keyword
        if features.get('has_suspicious_keyword'):
            score += 0.15

        # Brand impersonation
        if features.get('has_brand_name'):
            score += 0.2

        # @ symbol
        if features.get('has_at_symbol'):
            score += 0.25

        # Excessive subdomains
        if features.get('subdomain_count', 0) > 3:
            score += 0.15

        # Non-ASCII
        if features.get('has_non_ascii'):
            score += 0.2

        # Freenom domain
        if features.get('has_freenom_domain'):
            score += 0.15

        # Dash in domain
        if features.get('has_dash_in_domain'):
            score += 0.1

        return min(score, 1.0)

    def get_risk_level(self, confidence: float) -> str:
        """Convert confidence score to risk level"""
        from app.config import RISK_THRESHOLDS

        if confidence >= RISK_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        elif confidence >= RISK_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif confidence >= RISK_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
