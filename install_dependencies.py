import subprocess
import sys
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    basic_deps = [
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "evaluate",
        "jiwer",
        "soundfile",
        "gradio",
        "numpy",
        "matplotlib",
        "tqdm"
    ]

    for package in basic_deps:
        install_package(package)

    print("All dependencies installed successfully!")

if __name__ == "__main__":
    main()
