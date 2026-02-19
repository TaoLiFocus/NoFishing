# -*- coding: utf-8 -*-
"""
NoFishing ML API - Simplified Routes
简化路由 - 只做URL级别检测
"""
import logging
import time
from datetime import datetime

from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS

from app.config import *
from models.url_classifier import get_classifier

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create API blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')


@api_v1.route('/classify', methods=['POST'])
def classify():
    """
    对URL进行分类（纯URL词法特征检测）

    Request JSON:
    {
        "url": "http://example.com"
    }

    Response JSON:
    {
        "url": "http://example.com",
        "is_phishing": false,
        "probability": 0.15,
        "risk_level": "LOW",
        "confidence": 0.70,
        "features": {
            "url_length": 25,
            "domain_length": 15,
            ...
        },
        "processing_time_ms": 5,
        "timestamp": "2025-01-01T00:00:00Z"
    }

    注意：
    - probability: 钓鱼概率 (0-1)，越高越可能是钓鱼
    - risk_level: RISK_LEVEL (CRITICAL/HIGH/MEDIUM/LOW/SAFE)
    - Java后端可基于这些值决定是否需要深度分析
    """
    try:
        # Parse request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "error": "Missing required field: url"
            }), 400

        url = data['url']

        logger.info(f"Classifying URL: {url}")

        # Get classifier and run prediction
        classifier = get_classifier()
        result = classifier.classify(url)

        logger.info(f"Classification complete: is_phishing={result['is_phishing']}, "
                   f"probability={result['probability']:.3f}, "
                   f"risk_level={result['risk_level']}, "
                   f"time={result['processing_time_ms']}ms")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@api_v1.route('/batch_classify', methods=['POST'])
def batch_classify():
    """
    批量分类多个URL

    Request JSON:
    {
        "urls": ["http://example1.com", "http://example2.com", ...]
    }

    Response JSON:
    {
        "results": [
            {
                "url": "http://example1.com",
                "is_phishing": false,
                ...
            },
            ...
        ],
        "total_count": 2,
        "phishing_count": 1,
        "processing_time_ms": 10
    }
    """
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({
                "error": "Missing required field: urls"
            }), 400

        urls = data['urls']

        if len(urls) > 100:
            return jsonify({
                "error": "Too many URLs (maximum 100 per request)"
            }), 400

        logger.info(f"Batch classifying {len(urls)} URLs")

        classifier = get_classifier()
        start_time = time.time()

        results = []
        phishing_count = 0

        for url in urls:
            result = classifier.classify(url)
            results.append(result)

            if result['is_phishing']:
                phishing_count += 1

        total_time = int((time.time() - start_time) * 1000)

        response = {
            "results": results,
            "total_count": len(results),
            "phishing_count": phishing_count,
            "processing_time_ms": total_time
        }

        logger.info(f"Batch classification complete: {phishing_count}/{len(urls)} phishing, "
                   f"time={total_time}ms")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during batch classification: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@api_v1.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    try:
        from models.url_classifier import get_classifier
        classifier = get_classifier()

        return jsonify({
            "status": "healthy",
            "service": "nofishing-ml-api",
            "version": "1.0.0",
            "model_type": "url_lexical",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@api_v1.route('/ready', methods=['GET'])
def ready():
    """就绪探针"""
    return jsonify({"status": "ready"}), 200


# Register blueprints
app.register_blueprint(api_v1)

# Root endpoint
@app.route('/')
def index():
    return jsonify({
        "service": "NoFishing ML API",
        "version": "1.0.0",
        "model_type": "URL Lexical Feature Classifier",
        "description": "Fast phishing detection using URL lexical features only",
        "endpoints": {
            "classify": "POST /api/v1/classify",
            "batch_classify": "POST /api/v1/batch_classify",
            "health": "GET /api/v1/health",
            "ready": "GET /api/v1/ready"
        }
    }), 200


if __name__ == '__main__':
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
