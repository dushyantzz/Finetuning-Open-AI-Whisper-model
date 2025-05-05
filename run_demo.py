import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run ASR demo")
    parser.add_argument("--model_path", type=str, default="./models/whisper-fine-tuned-final")
    parser.add_argument("--simple", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Using the simple demo with the default Whisper model")
        args.simple = True

    if args.simple:
        cmd = ["python", "simple_demo.py"]
    else:
        cmd = ["python", "app/demo.py"]

    print(f"Running command: {' '.join(cmd)}")

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
