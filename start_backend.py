#!/usr/bin/env python3
"""
Startup script for the LAMDA Supply Chain Analysis Backend
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import fastapi
        import uvicorn
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r backend/requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting LAMDA Supply Chain Analysis Backend...")
    
    # Change to the backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("✗ Backend directory not found!")
        return False
    
    # Copy the model file to backend directory if it exists in root
    model_file = Path("tgn_model.pth")
    backend_model = backend_dir / "tgn_model.pth"
    
    if model_file.exists() and not backend_model.exists():
        print("Copying model file to backend directory...")
        import shutil
        shutil.copy2(model_file, backend_model)
    
    if not backend_model.exists():
        print("Warning: Model file not found. Backend will start but model predictions will be simulated.")
    
    try:
        # Start the FastAPI server
        os.chdir(backend_dir)
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nBackend server stopped")
    except Exception as e:
        print(f"✗ Error starting backend: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("LAMDA Supply Chain Analysis System")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    start_backend()