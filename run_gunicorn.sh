#!/bin/bash

# Navigate to the application directory if needed
# cd /path/to/your/app

# Run Gunicorn with binding options
echo "Starting Gunicorn..."
gunicorn --bind 0.0.0.0:8000 app:app --timeout 180