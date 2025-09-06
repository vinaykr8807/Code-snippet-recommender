#!/usr/bin/env python3
"""
Launcher script for the Code Snippet Recommendation System
This script starts the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("🚀 Starting CodeRec - Code Snippet Recommendation System")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if required files exist
    required_files = [
        "streamlit_app.py",
        "data_python.csv",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("✅ All required files found")
    print("🌐 Starting web application...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 60)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
