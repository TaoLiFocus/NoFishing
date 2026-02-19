# NoFishing - Phishing Website Detection System

Real-time phishing website detection and protection system using machine learning.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Chrome Extension│────▶│ Spring Boot API │────▶│  Flask + ML API │
│   (Manifest V3) │◀────│   (Java 17)     │◀────│  (PyTorch)      │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                        │
                                 └────────┬───────────────┘
                                          ▼
                                   ┌─────────────────┐
                                   │  Redis Cache    │
                                   └─────────────────┘
```

## Components

### 1. Chrome Extension (`nofishing-extension/`)
- Real-time URL interception
- Phishing warnings
- Safe browsing indicators
- Manifest V3 compatible

### 2. Spring Boot Backend (`nofishing-backend/`)
- REST API gateway
- Request caching with Redis
- ML service client
- Feature extraction

### 3. Flask ML API (`nofishing-ml-api/`)
- PyTorch neural network
- URL feature extraction
- Content analysis
- ONNX export support

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Java 17 (for local backend development)
- Python 3.11 (for local ML development)
- Node.js 18+ (for extension development)

### Using Docker Compose

```bash
# Clone repository
git clone https://github.com/yourusername/nofishing.git
cd nofishing

# Start all services
docker-compose up -d

# Verify services
curl http://localhost:8080/api/v1/health
curl http://localhost:5000/api/v1/health
```

### Local Development

#### Backend (Spring Boot)

```bash
cd nofishing-backend
mvn spring-boot:run
```

#### ML API (Flask)

```bash
cd nofishing-ml-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/routes.py
```

#### Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `nofishing-extension` directory

## API Endpoints

### Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect` | POST | Detect phishing URL |
| `/api/v1/check` | GET | Quick check URL |
| `/api/v1/health` | GET | Health check |

### ML API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/classify` | POST | Classify URL |
| `/api/v1/features/<url>` | GET | Extract features |
| `/api/v1/health` | GET | Health check |

## Usage Examples

### Detect a URL

```bash
curl -X POST http://localhost:8080/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com"}'
```

Response:
```json
{
  "url": "http://example.com",
  "isPhishing": false,
  "confidence": 0.1,
  "riskLevel": "LOW",
  "processingTimeMs": 45,
  "fromCache": false
}
```

### Quick Check

```bash
curl "http://localhost:8080/api/v1/check?url=http://suspicious-site.com"
```

## Training the Model

```bash
cd nofishing-ml-api/training

# Generate sample dataset
python dataset.py

# Train model
python train.py

# Evaluate model
python evaluate.py

# Export to ONNX
python ../models/onnx_exporter.py
```

## Configuration

### Backend (`application.yml`)

```yaml
ml-service:
  base-url: http://localhost:5000/api/v1
  timeout: 3000

nofishing:
  cache:
    ttl: 3600  # 1 hour
  detection:
    threshold: 0.5
```

### ML API (`config.py`)

```python
MODEL_TYPE = 'pytorch'  # or 'onnx'
USE_GPU = False
PHISHING_THRESHOLD = 0.5
```

## Performance Targets

| Component | Target Latency |
|-----------|----------------|
| URL → Backend | < 50ms |
| Backend → ML API | < 100ms |
| ML Inference | < 30ms |
| **End-to-End** | **< 200ms** |

## Project Structure

```
NoFishing/
├── nofishing-backend/          # Spring Boot API
│   ├── src/main/java/
│   │   └── com/nofishing/
│   │       ├── controller/     # REST endpoints
│   │       ├── service/        # Business logic
│   │       ├── client/         # ML service client
│   │       ├── dto/            # Data transfer objects
│   │       ├── config/         # Configuration
│   │       └── exception/      # Error handling
│   └── pom.xml
│
├── nofishing-ml-api/          # Flask ML API
│   ├── app/
│   │   ├── routes.py          # API endpoints
│   │   ├── config.py         # Configuration
│   │   └── utils/            # Utilities
│   ├── models/
│   │   ├── phishing_classifier.py
│   │   └── onnx_exporter.py
│   ├── training/
│   │   ├── train.py
│   │   ├── dataset.py
│   │   └── evaluate.py
│   └── requirements.txt
│
├── nofishing-extension/       # Chrome Extension
│   ├── manifest.json
│   ├── src/
│   │   ├── background/        # Service worker
│   │   ├── content/          # Content scripts
│   │   ├── popup/            # Popup UI
│   │   └── utils/            # Utilities
│   └── public/
│       └── warning.html       # Warning page
│
└── docker-compose.yml
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for educational purposes. Always exercise caution when browsing the internet.
