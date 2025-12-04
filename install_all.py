# install_all.py - Run this to install everything
import subprocess
import sys

def install_packages():
    packages = [
        "setuptools",
        "wheel", 
        "pretty-midi",
        "mido",
        "python-rtmidi",
        "numpy",
        "scipy",
        "fastdtw",
        "matplotlib",
        "pandas"
    ]
    
    print("=" * 60)
    print("INSTALLING ALL DEPENDENCIES")
    print("=" * 60)
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            print("Trying with --user flag...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
                print(f"✓ {package} installed with --user flag")
            except:
                print(f"✗ Could not install {package}")
    
    print("\n" + "=" * 60)
    print("VERIFYING INSTALLATION")
    print("=" * 60)
    
    # Test imports
    test_imports = [
        ("setuptools", "import setuptools"),
        ("pretty_midi", "import pretty_midi"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("fastdtw", "from fastdtw import fastdtw"),
    ]
    
    for name, import_cmd in test_imports:
        print(f"\nTesting {name}...")
        try:
            exec(import_cmd)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")

if __name__ == "__main__":
    install_packages()
    print("\n✅ Installation complete!")
    print("\nNow run: python examples/demo_usage.py")