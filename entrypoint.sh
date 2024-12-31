#!/bin/bash

if [ "$POLLING" = "true" ]; then
    echo "POLLING is true. Starting python app.py..."
    python app.py
else
    echo "POLLING is not true. Starting Gunicorn server..."
    gunicorn -w 4 --timeout 1000 -b 0.0.0.0:8443 'app:app'
fi
