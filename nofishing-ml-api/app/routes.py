"""
NoFishing ML API - Flask Routes
"""
import logging
import time
from datetime import datetime

from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS

from app.config import *

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

# Lazy load classifier to avoid import errors
_classifier = None


def get_classifier():
    """Lazy load the classifier"""
    global _classifier
    if _classifier is None:
        from app.models.phishing_classifier import get_classifier as get_model
        _classifier = get_model()
        logger.info("Classifier initialized")
    return _classifier


@api_v1.route('/classify', methods=['POST'])
def classify():
    """
    Classify a URL as phishing or legitimate

    Request JSON:
    {
        "url": "http://example.com",
        "fetch_content": true  // optional
    }

    Response JSON:
    {
        "is_phishing": false,
        "probability": 0.95,
        "risk_level": "LOW",
        "features": {...},
        "processing_time_ms": 150
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "error": "Missing required field: url"
            }), 400

        url = data['url']
        fetch_content = data.get('fetch_content', False)

        logger.info(f"Classifying URL: {url}")

        # Get classifier and run prediction
        classifier = get_classifier()
        start_time = time.time()

        result = classifier.predict(url, fetch_content=fetch_content)

        processing_time = int((time.time() - start_time) * 1000)
        result['processing_time_ms'] = processing_time

        logger.info(f"Classification complete: is_phishing={result['is_phishing']}, "
                   f"probability={result['confidence']:.3f}, time={processing_time}ms")

        # Map 'confidence' to 'probability' for Java backend compatibility
        response = {
            'is_phishing': result['is_phishing'],
            'probability': result.get('confidence', 0.0),
            'risk_level': result.get('risk_level', 'LOW'),
            'features': result.get('features', {}),
            'processing_time_ms': processing_time
        }

        # Include error if present
        if 'error' in result:
            response['error'] = result['error']

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@api_v1.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Check if classifier is loaded
        classifier = get_classifier()

        return jsonify({
            "status": "healthy",
            "service": "nofishing-ml-api",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": classifier is not None,
            "model_type": MODEL_TYPE
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@api_v1.route('/ready', methods=['GET'])
def ready():
    """Readiness probe"""
    return jsonify({"status": "ready"}), 200


@api_v1.route('/features/<path:url>', methods=['GET'])
def extract_features(url):
    """
    Extract features from a URL without classification

    Useful for debugging and feature analysis
    """
    try:
        from utils.url_processor import URLProcessor

        processor = URLProcessor()
        features = processor.extract_features(url)

        return jsonify({
            "url": url,
            "features": features
        }), 200

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# Register blueprints
app.register_blueprint(api_v1)

# Root endpoint
@app.route('/')
def index():
    return jsonify({
        "service": "NoFishing ML API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "POST /api/v1/classify",
            "health": "GET /api/v1/health",
            "features": "GET /api/v1/features/<url>"
        }
    }), 200


if __name__ == '__main__':
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
