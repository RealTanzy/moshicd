#!/bin/bash

# Creative Dashboard Launch Script
# This script helps you launch the Creative Dashboard application

echo "🎨 Creative Dashboard Launcher"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip found"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please ensure all files are in the current directory."
    exit 1
fi

echo "✅ requirements.txt found"

# Check if main application exists
if [ ! -f "creative_dashboard.py" ]; then
    echo "❌ creative_dashboard.py not found. Please ensure all files are in the current directory."
    exit 1
fi

echo "✅ creative_dashboard.py found"


# Launch the application
echo ""
echo "🚀 Launching Creative Dashboard..."
echo "📍 The application will open in your default browser"
echo "🌐 URL: http://localhost:8501"
echo ""
echo "💡 Tip: Keep this terminal window open while using the application"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run creative_dashboard.py

echo ""
echo "👋 Creative Dashboard has been stopped"
echo "📝 Thank you for using Creative Dashboard!"
