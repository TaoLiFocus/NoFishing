# -*- coding: utf-8 -*-
"""Test feature count"""

import re
from urllib.parse import urlparse

url = "https://www.google.com/search"
url_lower = url.lower()
parsed = urlparse(url)
hostname = parsed.hostname or ''

features = [
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
    0,                                  # 26
    1,                                  # 27
    0,                                  # 28
    0,                                  # 29
    0,                                  # 30
]

print(f"Total features: {len(features)}")
