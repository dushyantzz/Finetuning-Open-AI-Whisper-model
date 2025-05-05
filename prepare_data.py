import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for ASR training")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/processed_common_voice")
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--no_augmentation", action="store_true")

    args = parser.parse_args()

    cmd = [
        "python", "scripts/data_preparation.py",
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--max_files", str(args.max_files),
        "--batch_size", str(args.batch_size)
    ]

    if args.transcribe:
        cmd.append("--transcribe")

    if args.no_augmentation:
        cmd.append("--no_augmentation")

    print(f"Running command: {' '.join(cmd)}")

    start_time = time.time()
    subprocess.run(cmd)
    elapsed_time = time.time() - start_time

    print(f"Data preparation completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
