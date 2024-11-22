#!/bin/bash

# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/pcam
mkdir -p logs/checkpoints
mkdir -p results

# Make the script executable
chmod +x benchmark.py

echo "Setup completed successfully!"
echo "To start benchmarking, run: python benchmark.py"
