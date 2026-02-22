"""
Quick start script for SR-MARE web interface.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies installed")
        return True
    except ImportError:
        print("❌ FastAPI dependencies not found")
        print("\nInstalling required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "jinja2", "python-multipart"])
        print("✅ Dependencies installed")
        return True

def main():
    print("=" * 60)
    print("🚀 SR-MARE Web Interface Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the server
    print("\n🌐 Starting FastAPI server...")
    print("📱 Open your browser at: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
