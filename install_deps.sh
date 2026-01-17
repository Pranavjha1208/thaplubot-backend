#!/bin/bash
# Install python-dotenv in the virtual environment

cd /Users/pranav/pj/latest/backend8
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Dependencies installed successfully!"
