#!/bin/bash
# NoFishing ML API - Startup Script

set -e

# Configuration
FLASK_APP=${FLASK_APP:-app.routes}
FLASK_ENV=${FLASK_ENV:-production}
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
WORKERS=${GUNICORN_WORKERS:-2}

echo "Starting NoFishing ML API..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run with gunicorn
exec gunicorn \
    --bind $HOST:$PORT \
    --workers $WORKERS \
    --worker-class sync \
    --timeout 30 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --config gunicorn_config.py \
    "$FLASK_APP:app"
