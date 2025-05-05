# run_pipeline.py
import os
import argparse
import logging
import subprocess
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the output"""
    logger.info(f"Running {description}...")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )

        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())

        # Get return code
        return_code = process.poll()

        if return_code == 0:
            logger.info(f"{description} completed successfully")
            return True
        else:
            error = process.stderr.read()
            logger.error(f"{description} failed with return code {return_code}")
            logger.error(f"Error: {error}")
            return False

    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return False

def main():
    """Run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run the complete ASR pipeline")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the data directory")
    parser.add_argument("--max_files", type=int, default=50,
                        help="Maximum number of files to process")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip model training step")
    parser.add_argument("--skip_error_correction", action="store_true",
                        help="Skip error correction training")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--run_demo", action="store_true",
                        help="Run the demo after pipeline completion")
    parser.add_argument("--transcribe", action="store_true",
                        help="Transcribe audio files using Whisper")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and transcription")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny",
                        help="Name or path of the base model (use tiny for faster testing)")

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("./data/processed_common_voice", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Step 1: Data Preparation
    if not args.skip_data_prep:
        data_prep_cmd = (
            f"python prepare_data.py "
            f"--data_path {args.data_path} "
            f"--max_files {args.max_files} "
            f"--batch_size {args.batch_size}"
        )

        if args.transcribe:
            data_prep_cmd += " --transcribe"

        if not run_command(data_prep_cmd, "Data preparation"):
            logger.error("Data preparation failed. Exiting pipeline.")
            return

    # Step 2: Model Training
    if not args.skip_training:
        train_cmd = (
            f"python train_model.py "
            f"--model_name {args.model_name} "
            f"--num_epochs {args.num_epochs} "
            f"--batch_size {args.batch_size}"
        )

        if not run_command(train_cmd, "Model training"):
            logger.warning("Model training failed. Continuing with pipeline.")

    # Step 3: Error Correction Training
    if not args.skip_error_correction:
        error_correction_cmd = (
            f"python scripts/error_correction.py "
            f"--output_dir ./models/error_corrector "
            f"--num_epochs {args.num_epochs} "
            f"--example"
        )

        if not run_command(error_correction_cmd, "Error correction model training"):
            logger.warning("Error correction model training failed. Continuing with pipeline.")

    # Step 4: Evaluation
    if not args.skip_evaluation:
        eval_cmd = (
            f"python scripts/evaluation.py "
            f"--model_path ./models/whisper-fine-tuned-final "
            f"--dataset_path ./data/processed_common_voice "
            f"--max_samples {min(args.max_files // 2, 20)}"
        )

        if not run_command(eval_cmd, "Model evaluation"):
            logger.warning("Model evaluation failed. Continuing with pipeline.")

    # Step 5: Run Demo
    if args.run_demo:
        demo_cmd = "python run_demo.py --simple"
        if not run_command(demo_cmd, "Interactive demo"):
            logger.error("Failed to launch demo.")

    logger.info("Pipeline completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total pipeline execution time: {elapsed_time:.2f} seconds")
