#!/bin/bash

# Quick setup script for the churn prediction platform

echo "🎯 Setting up Customer Churn Prediction Platform..."
echo ""

# Create virtual environment
echo "1️⃣  Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "2️⃣  Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "3️⃣  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "4️⃣  Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "5️⃣  Creating directory structure..."
mkdir -p data/raw logs

# Check if telco.csv exists
echo "6️⃣  Checking for dataset..."
if [ -f "telco.csv" ]; then
    echo "   ✓ Found telco.csv, moving to data/raw/"
    mv telco.csv data/raw/
elif [ -f "data/raw/telco.csv" ]; then
    echo "   ✓ Dataset already in place"
else
    echo "   ⚠️  Dataset not found. Please download IBM Telco Customer Churn dataset"
    echo "      and place it in data/raw/telco.csv"
    echo ""
    echo "      Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download the Telco dataset if you haven't already"
echo "  2. Run the full pipeline:"
echo "     python run_pipeline.py --mode full"
echo ""
echo "  Or run individual steps:"
echo "     python -m src.data_ingestion.run"
echo "     python -m src.training_pipeline.run"
echo "     python -m src.inference.batch"
echo ""
echo "  Start the API server:"
echo "     uvicorn src.inference.api:app --reload"
echo ""
