"""
Quick setup verification script.
Run this after installation to verify everything is working.
"""

import sys
import subprocess


def check_python_version():
    """Check if Python version is sufficient."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False


def check_ollama_installed():
    """Check if Ollama is installed."""
    print("\n🦙 Checking Ollama installation...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✅ Ollama is installed")
            return True, result.stdout
        else:
            print("   ❌ Ollama command failed")
            return False, ""
    except FileNotFoundError:
        print("   ❌ Ollama not found in PATH")
        print("   Install from: https://ollama.ai")
        return False, ""
    except Exception as e:
        print(f"   ❌ Error checking Ollama: {e}")
        return False, ""


def check_models(ollama_output):
    """Check if required models are installed."""
    print("\n🤖 Checking required models...")
    
    required_models = ["mistral", "llama3.2", "nomic-embed-text"]
    installed_models = ollama_output.lower()
    
    all_installed = True
    for model in required_models:
        if model in installed_models:
            print(f"   ✅ {model}")
        else:
            print(f"   ❌ {model} (not installed)")
            print(f"      Run: ollama pull {model}")
            all_installed = False
    
    return all_installed


def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n📦 Checking Python packages...")
    
    required_packages = {
        "requests": "requests",
        "numpy": "numpy",
        "faiss": "faiss-cpu"
    }
    
    all_installed = True
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (not installed)")
            all_installed = False
    
    if not all_installed:
        print("\n   Install missing packages:")
        print("   pip install -r requirements.txt")
    
    return all_installed


def test_ollama_connection():
    """Test connection to Ollama API."""
    print("\n🔌 Testing Ollama API connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama API is accessible")
            return True
        else:
            print(f"   ❌ Ollama API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to Ollama API")
        print("   Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SR-MARE SETUP VERIFICATION")
    print("=" * 60)
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    
    # Ollama installation
    ollama_installed, ollama_output = check_ollama_installed()
    checks.append(ollama_installed)
    
    # Models
    if ollama_installed:
        checks.append(check_models(ollama_output))
    else:
        print("\n⚠️  Skipping model check (Ollama not installed)")
        checks.append(False)
    
    # Python packages
    checks.append(check_python_packages())
    
    # API connection
    checks.append(test_ollama_connection())
    
    # Summary
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou're ready to use SR-MARE!")
        print("\nTry running:")
        print("  python main.py --test-connection")
        print("  python example.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before using SR-MARE.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
