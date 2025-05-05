import os
import argparse
import logging
import json
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from tqdm import tqdm
import evaluate
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error_correction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASRErrorCorrector:
    def __init__(self, model_name="t5-small", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing ASR Error Corrector with model {model_name} on {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def prepare_dataset(self, asr_outputs, ground_truth, prefix="correct: "):
        logger.info(f"Preparing dataset with {len(asr_outputs)} examples")

        dataset = Dataset.from_dict({
            "asr_output": asr_outputs,
            "ground_truth": ground_truth
        })

        def preprocess_function(examples):
            inputs = [prefix + text for text in examples["asr_output"]]
            targets = examples["ground_truth"]

            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")

            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_dataset = dataset.map(preprocess_function, batched=True)
        logger.info(f"Dataset prepared with {len(processed_dataset)} examples")

        return processed_dataset

    def train(self, train_dataset, eval_dataset=None, output_dir="./models/error_corrector",
              batch_size=8, learning_rate=5e-5, num_epochs=3, save_steps=500):
        logger.info(f"Training error correction model with {len(train_dataset)} examples")

        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_strategy="steps" if save_steps > 0 else "epoch",
            save_steps=save_steps if save_steps > 0 else 500,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if save_steps > 0 else 500,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to=["tensorboard"],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info("Starting training")
        train_result = trainer.train()

        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2)

        logger.info(f"Training results: {train_result}")

        if eval_dataset:
            logger.info("Evaluating model")
            eval_result = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_result}")

            with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_result, f, indent=2)

        return train_result

    def correct(self, asr_output, max_length=512, num_beams=5, prefix="correct: "):
        input_text = prefix + asr_output
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def evaluate_corrections(self, asr_outputs, ground_truth):
        logger.info(f"Evaluating error correction on {len(asr_outputs)} examples")

        metrics = {
            "original": {
                "wer": [],
                "cer": []
            },
            "corrected": {
                "wer": [],
                "cer": []
            }
        }

        corrected_outputs = []
        for asr_output, reference in tqdm(zip(asr_outputs, ground_truth), total=len(asr_outputs), desc="Evaluating"):
            wer_original = self.wer_metric.compute(predictions=[asr_output], references=[reference])
            cer_original = self.cer_metric.compute(predictions=[asr_output], references=[reference])

            metrics["original"]["wer"].append(wer_original)
            metrics["original"]["cer"].append(cer_original)

            corrected = self.correct(asr_output)
            corrected_outputs.append(corrected)

            wer_corrected = self.wer_metric.compute(predictions=[corrected], references=[reference])
            cer_corrected = self.cer_metric.compute(predictions=[corrected], references=[reference])

            metrics["corrected"]["wer"].append(wer_corrected)
            metrics["corrected"]["cer"].append(cer_corrected)

        avg_metrics = {
            "original": {
                "wer": np.mean(metrics["original"]["wer"]),
                "cer": np.mean(metrics["original"]["cer"])
            },
            "corrected": {
                "wer": np.mean(metrics["corrected"]["wer"]),
                "cer": np.mean(metrics["corrected"]["cer"])
            }
        }

        avg_metrics["improvement"] = {
            "wer": avg_metrics["original"]["wer"] - avg_metrics["corrected"]["wer"],
            "cer": avg_metrics["original"]["cer"] - avg_metrics["corrected"]["cer"]
        }

        logger.info(f"Original WER: {avg_metrics['original']['wer']:.4f}")
        logger.info(f"Corrected WER: {avg_metrics['corrected']['wer']:.4f}")
        logger.info(f"WER Improvement: {avg_metrics['improvement']['wer']:.4f}")
        logger.info(f"Original CER: {avg_metrics['original']['cer']:.4f}")
        logger.info(f"Corrected CER: {avg_metrics['corrected']['cer']:.4f}")
        logger.info(f"CER Improvement: {avg_metrics['improvement']['cer']:.4f}")

        return avg_metrics, corrected_outputs

    def create_error_dataset(self, texts, error_rate=0.1, error_types=None):
        if error_types is None:
            error_types = ["swap", "delete", "insert", "replace", "split", "join"]

        import random
        import string

        def introduce_errors(text):
            words = text.split()
            result = []

            i = 0
            while i < len(words):
                word = words[i]

                if random.random() < error_rate and len(word) > 1:
                    error_type = random.choice(error_types)

                    if error_type == "swap" and len(word) > 2:
                        pos = random.randint(0, len(word) - 2)
                        word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

                    elif error_type == "delete":
                        pos = random.randint(0, len(word) - 1)
                        word = word[:pos] + word[pos+1:]

                    elif error_type == "insert":
                        pos = random.randint(0, len(word))
                        char = random.choice(string.ascii_lowercase)
                        word = word[:pos] + char + word[pos:]

                    elif error_type == "replace":
                        pos = random.randint(0, len(word) - 1)
                        char = random.choice(string.ascii_lowercase)
                        word = word[:pos] + char + word[pos+1:]

                    elif error_type == "split" and len(word) > 3 and i < len(words) - 1:
                        pos = random.randint(1, len(word) - 1)
                        result.append(word[:pos])
                        word = word[pos:]

                    elif error_type == "join" and i < len(words) - 1:
                        word = word + words[i+1]
                        i += 1

                result.append(word)
                i += 1

            return " ".join(result)

        texts_with_errors = [introduce_errors(text) for text in texts]

        return texts_with_errors, texts

def load_asr_dataset(dataset_path=None, split="test", max_samples=1000):
    logger.info(f"Loading ASR dataset from {dataset_path or 'Common Voice'}")

    if dataset_path:
        from datasets import load_from_disk
        try:
            dataset = load_from_disk(os.path.join(dataset_path, split))
            logger.info(f"Loaded {len(dataset)} examples from {os.path.join(dataset_path, split)}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None, None
    else:
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split=split)
            logger.info(f"Loaded {len(dataset)} examples from Common Voice")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None, None

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {len(dataset)} examples")

    ground_truth = dataset["sentence"]

    corrector = ASRErrorCorrector()
    asr_outputs, _ = corrector.create_error_dataset(ground_truth, error_rate=0.2)

    return asr_outputs, ground_truth

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ASR error correction model")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/error_corrector")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--example", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    corrector = ASRErrorCorrector(model_name=args.model_name)

    if args.example:
        logger.info("Using example data")

        asr_outputs = [
            "this is an exemple of incorrect transscription",
            "the wether is nice today",
            "she lives in new yok city",
            "i need to book a fligt to sanfrancisco",
            "the meating is schedled for tommorow",
            "he recieved an importent email",
            "the resturant is open untill midnight",
            "she studys mathemetics at the uniersity",
            "the concert was cancled due to wether",
            "they're planing a trip to europ next summer"
        ]

        ground_truth = [
            "this is an example of incorrect transcription",
            "the weather is nice today",
            "she lives in new york city",
            "i need to book a flight to san francisco",
            "the meeting is scheduled for tomorrow",
            "he received an important email",
            "the restaurant is open until midnight",
            "she studies mathematics at the university",
            "the concert was canceled due to weather",
            "they're planning a trip to europe next summer"
        ]

        train_size = int(0.8 * len(asr_outputs))
        train_asr = asr_outputs[:train_size]
        train_gt = ground_truth[:train_size]
        eval_asr = asr_outputs[train_size:]
        eval_gt = ground_truth[train_size:]

    else:
        asr_outputs, ground_truth = load_asr_dataset(
            dataset_path=args.dataset_path,
            max_samples=args.max_samples
        )

        if not asr_outputs:
            logger.error("Failed to load dataset")
            return

        train_size = int(0.8 * len(asr_outputs))
        train_asr = asr_outputs[:train_size]
        train_gt = ground_truth[:train_size]
        eval_asr = asr_outputs[train_size:]
        eval_gt = ground_truth[train_size:]

    if not args.eval_only:
        train_dataset = corrector.prepare_dataset(train_asr, train_gt)
        eval_dataset = corrector.prepare_dataset(eval_asr, eval_gt)

        corrector.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )

    if os.path.exists(args.output_dir):
        logger.info(f"Loading trained model from {args.output_dir}")
        corrector = ASRErrorCorrector(model_name=args.output_dir)

    metrics, corrected_outputs = corrector.evaluate_corrections(eval_asr, eval_gt)

    logger.info("\nExamples:")
    for i in range(min(5, len(eval_asr))):
        logger.info(f"Original: {eval_asr[i]}")
        logger.info(f"Corrected: {corrected_outputs[i]}")
        logger.info(f"Ground truth: {eval_gt[i]}")
        logger.info("---")

    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.json')}")

if __name__ == "__main__":
    main()
