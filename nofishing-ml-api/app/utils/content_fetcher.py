"""
Content Fetcher - Fetches and analyzes web page content
"""
import logging
import re
from typing import Dict, Optional, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import CONTENT_FETCH_TIMEOUT

logger = logging.getLogger(__name__)


class ContentFetcher:
    """Fetches and analyzes web page content for phishing indicators"""

    # Suspicious form action patterns
    SUSPICIOUS_FORM_ACTIONS = [
        r'login', r'signin', r'auth', r'account', r'verify',
        r'password', r'credential', r'bank'
    ]

    # Input types that suggest credential theft
    CREDENTIAL_INPUT_TYPES = {'password', 'pin', 'ssn', 'card'}

    # User agent for requests
    USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )

    def __init__(self, timeout: int = CONTENT_FETCH_TIMEOUT):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})
        self.logger = logger

    def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and analyze page content

        Args:
            url: URL to fetch

        Returns:
            Dictionary with content features
        """
        try:
            # Validate URL first
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {'error': 'invalid_url'}

            # Fetch the page
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                verify=True  # SSL verification
            )

            if response.status_code != 200:
                return {
                    'error': 'fetch_failed',
                    'status_code': response.status_code
                }

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract features
            features = {
                'status_code': response.status_code,
                'content_length': len(response.text),
                'redirected': response.history is not None and len(response.history) > 0,
                'redirect_count': len(response.history) if response.history else 0,
                'final_url': response.url
            }

            # Add HTML features
            features.update(self._extract_html_features(soup))

            # Add security features
            features.update(self._extract_security_features(response, url))

            self.logger.info(f"Successfully fetched content from {url}")
            return features

        except requests.exceptions.Timeout:
            return {'error': 'timeout'}
        except requests.exceptions.SSLError:
            return {'error': 'ssl_error'}
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to fetch {url}: {e}")
            return {'error': 'fetch_failed', 'message': str(e)}
        except Exception as e:
            self.logger.error(f"Error processing content from {url}: {e}")
            return {'error': 'processing_error', 'message': str(e)}

    def _extract_html_features(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract features from HTML content"""
        features = {}

        # Count forms
        forms = soup.find_all('form')
        features['form_count'] = len(forms)

        # Count inputs
        inputs = soup.find_all('input')
        features['input_count'] = len(inputs)

        # Check for password inputs
        password_inputs = soup.find_all('input', {'type': 'password'})
        features['password_input_count'] = len(password_inputs)

        # Check for credential inputs
        credential_input_count = sum(
            1 for inp in inputs
            if inp.get('name', '').lower() in self.CREDENTIAL_INPUT_TYPES or
               inp.get('type', '').lower() in self.CREDENTIAL_INPUT_TYPES
        )
        features['credential_input_count'] = credential_input_count

        # Analyze form actions
        suspicious_form_count = 0
        for form in forms:
            action = form.get('action', '')
            if any(re.search(pattern, action, re.IGNORECASE)
                   for pattern in self.SUSPICIOUS_FORM_ACTIONS):
                suspicious_form_count += 1

        features['suspicious_form_count'] = suspicious_form_count

        # Count links
        links = soup.find_all('a')
        features['link_count'] = len(links)

        # Count external links
        if forms:
            base_domain = urlparse(forms[0].get('action', '')).netloc
            external_links = sum(
                1 for link in links
                if link.get('href') and
                   urlparse(link['href']).netloc != base_domain
            )
            features['external_link_count'] = external_links
        else:
            features['external_link_count'] = 0

        # Check for images
        features['image_count'] = len(soup.find_all('img'))

        # Check for iframes (often used for phishing)
        features['iframe_count'] = len(soup.find_all('iframe'))

        # Check for scripts
        features['script_count'] = len(soup.find_all('script'))

        # Check for meta tags
        features['meta_count'] = len(soup.find_all('meta'))

        # Check for login indicators in text
        page_text = soup.get_text().lower()
        login_keywords = ['login', 'sign in', 'signin', 'auth', 'account']
        features['has_login_text'] = any(kw in page_text for kw in login_keywords)

        return features

    def _extract_security_features(self, response: requests.Response, url: str) -> Dict[str, Any]:
        """Extract security-related features"""
        features = {}

        # Check for HTTPS
        features['is_https'] = urlparse(url).scheme == 'https'

        # Check for security headers
        headers = response.headers
        features['has_hsts'] = 'Strict-Transport-Security' in headers
        features['has_csp'] = 'Content-Security-Policy' in headers
        features['has_xfo'] = 'X-Frame-Options' in headers

        # Check for cookies
        features['has_cookies'] = 'Cookie' in headers or 'Set-Cookie' in headers

        return features

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
