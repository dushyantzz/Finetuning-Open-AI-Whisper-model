# train_model.py
import os
import argparse
import subprocess
import time

def main():
    """Run the model training script with the specified options"""
    parser = argparse.ArgumentParser(description="Train ASR model")
    parser.add_argument("--dataset_path", type=str, default="./data/processed_common_voice",
                        help="Path to the processed dataset")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny",
                        help="Name or path of the base model (use tiny for faster testing)")
    parser.add_argument("--output_dir", type=str, default="./models/whisper-fine-tuned",
                        help="Directory to save the model")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    args = parser.parse_args()
    
    # Create command
    cmd = [
        "python", "scripts/model_training.py",
        "--dataset_path", args.dataset_path,
        "--model_name", args.model_name,
        "--output_dir", args.output_dir,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate)
    ]
    
    # Print command
    print(f"Running command: {' '.join(cmd)}")
    
    # Run command
    start_time = time.time()
    subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    print(f"Model training completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
