#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/dermmnist
mkdir -p logs/checkpoints
mkdir -p logs/wandb
mkdir -p results

# Make the script executable
chmod +x benchmark.py

echo "Setup completed successfully!"
echo "To start training, run: python benchmark.py"
