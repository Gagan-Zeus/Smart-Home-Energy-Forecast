#!/bin/bash

# Energy Consumption Forecasting App Launcher

echo "================================================"
echo "Energy Consumption Forecasting Web Application"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Check if data file exists
if [ ! -f "smart_home_energy_consumption_large.csv" ]; then
    echo ""
    echo "WARNING: Data file 'smart_home_energy_consumption_large.csv' not found!"
    echo "Please make sure the CSV file is in the current directory."
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Run the Streamlit app
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
streamlit run app.py
