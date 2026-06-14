#!/bin/bash
set -e

echo "Starting environment setup..."

# 1. Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# 2. Create the virtual environment named '.venv'
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'..."
    python3 -m venv .venv
else
    echo "Virtual environment '.venv' already exists."
fi

# 3. Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "Setup complete! You are ready to go."
echo "To activate the environment manually in the future, run: source .venv/bin/activate"
