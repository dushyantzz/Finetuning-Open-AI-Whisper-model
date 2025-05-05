import os
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
try:
    from error_correction import ASRErrorCorrector
except ImportError:
    print("Error correction module not found. Evaluation will be done without error correction.")
    ASRErrorCorrector = None

class ASRModelEvaluator:
    def __init__(self, model_path=None, processor_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.processor = WhisperProcessor.from_pretrained(processor_path or model_path)
        else:
            print("Loading pre-trained Whisper model (small)")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.bleu_metric = evaluate.load("bleu")
        self.error_corrector = None
        if ASRErrorCorrector:
            try:
                self.error_corrector = ASRErrorCorrector()
                error_model_path = "./models/error_corrector"
                if os.path.exists(error_model_path):
                    self.error_corrector.model = torch.load(f"{error_model_path}/pytorch_model.bin")
                    print("Loaded fine-tuned error correction model")
            except Exception as e:
                print(f"Failed to initialize error corrector: {e}")
                self.error_corrector = None

    def transcribe(self, audio_array, sampling_rate=16000, apply_correction=False):
        if sampling_rate != 16000:
            import torchaudio
            audio_array = torchaudio.functional.resample(
                torch.tensor(audio_array), sampling_rate, 16000).numpy()
            sampling_rate = 16000

        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if apply_correction and self.error_corrector:
            corrected_transcription = self.error_corrector.correct(transcription)
            return transcription, corrected_transcription

        return transcription, None

    def evaluate_dataset(self, dataset_path, split="test", apply_correction=True,
                         max_samples=None, dialect=None, output_file=None):
        try:
            dataset = load_from_disk(os.path.join(dataset_path, split))
            print(f"Loaded {len(dataset)} examples from {os.path.join(dataset_path, split)}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

        if dialect:
            dataset = dataset.filter(lambda example: example["accent"] == dialect)
            print(f"Filtered to {len(dataset)} examples with dialect '{dialect}'")

        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Limited evaluation to {len(dataset)} examples")

        results = {
            "raw": {
                "wer": [], "cer": [], "bleu": []
            },
            "corrected": {
                "wer": [], "cer": [], "bleu": []
            },
            "examples": []
        }

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            audio = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
            reference = example["sentence"]

            if not reference or reference.strip() == "":
                continue

            raw_transcription, corrected_transcription = self.transcribe(
                audio, sampling_rate, apply_correction=apply_correction
            )

            wer_raw = self.wer_metric.compute(predictions=[raw_transcription], references=[reference])
            cer_raw = self.cer_metric.compute(predictions=[raw_transcription], references=[reference])
            bleu_raw = self.bleu_metric.compute(predictions=[raw_transcription.split()], references=[[reference.split()]])

            results["raw"]["wer"].append(wer_raw)
            results["raw"]["cer"].append(cer_raw)
            results["raw"]["bleu"].append(bleu_raw["bleu"])

            if corrected_transcription:
                wer_corrected = self.wer_metric.compute(predictions=[corrected_transcription], references=[reference])
                cer_corrected = self.cer_metric.compute(predictions=[corrected_transcription], references=[reference])
                bleu_corrected = self.bleu_metric.compute(predictions=[corrected_transcription.split()], references=[[reference.split()]])

                results["corrected"]["wer"].append(wer_corrected)
                results["corrected"]["cer"].append(cer_corrected)
                results["corrected"]["bleu"].append(bleu_corrected["bleu"])

            example_result = {
                "id": i,
                "reference": reference,
                "raw_transcription": raw_transcription,
                "corrected_transcription": corrected_transcription,
                "wer_raw": wer_raw,
                "cer_raw": cer_raw,
                "bleu_raw": bleu_raw["bleu"]
            }

            if corrected_transcription:
                example_result.update({
                    "wer_corrected": wer_corrected,
                    "cer_corrected": cer_corrected,
                    "bleu_corrected": bleu_corrected["bleu"]
                })

            results["examples"].append(example_result)

        avg_results = {
            "raw": {
                "wer": np.mean(results["raw"]["wer"]),
                "cer": np.mean(results["raw"]["cer"]),
                "bleu": np.mean(results["raw"]["bleu"])
            }
        }

        if apply_correction and self.error_corrector:
            avg_results["corrected"] = {
                "wer": np.mean(results["corrected"]["wer"]),
                "cer": np.mean(results["corrected"]["cer"]),
                "bleu": np.mean(results["corrected"]["bleu"])
            }

            avg_results["improvement"] = {
                "wer": avg_results["raw"]["wer"] - avg_results["corrected"]["wer"],
                "cer": avg_results["raw"]["cer"] - avg_results["corrected"]["cer"],
                "bleu": avg_results["corrected"]["bleu"] - avg_results["raw"]["bleu"]
            }

        print("\nEvaluation Results:")
        print(f"Raw WER: {avg_results['raw']['wer']:.4f}")
        print(f"Raw CER: {avg_results['raw']['cer']:.4f}")
        print(f"Raw BLEU: {avg_results['raw']['bleu']:.4f}")

        if apply_correction and self.error_corrector:
            print(f"Corrected WER: {avg_results['corrected']['wer']:.4f}")
            print(f"Corrected CER: {avg_results['corrected']['cer']:.4f}")
            print(f"Corrected BLEU: {avg_results['corrected']['bleu']:.4f}")
            print(f"WER Improvement: {avg_results['improvement']['wer']:.4f}")
            print(f"CER Improvement: {avg_results['improvement']['cer']:.4f}")
            print(f"BLEU Improvement: {avg_results['improvement']['bleu']:.4f}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "average_metrics": avg_results,
                    "examples": results["examples"][:10]
                }, f, indent=2)
            print(f"Results saved to {output_file}")

        return avg_results, results["examples"]

    def evaluate_by_dialect(self, dataset_path, split="test", dialects=None,
                           apply_correction=True, max_samples_per_dialect=50):
        try:
            dataset = load_from_disk(os.path.join(dataset_path, split))
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

        if not dialects:
            dialects = set(dataset["accent"])

        dialect_results = {}

        for dialect in dialects:
            print(f"\nEvaluating dialect: {dialect}")
            metrics, examples = self.evaluate_dataset(
                dataset_path,
                split=split,
                apply_correction=apply_correction,
                max_samples=max_samples_per_dialect,
                dialect=dialect,
                output_file=f"./results/evaluation_{dialect}.json"
            )

            dialect_results[dialect] = metrics

        self.plot_dialect_comparison(dialect_results)

        return dialect_results

    def plot_dialect_comparison(self, dialect_results):
        dialects = list(dialect_results.keys())

        raw_wer = [dialect_results[d]["raw"]["wer"] for d in dialects]
        raw_cer = [dialect_results[d]["raw"]["cer"] for d in dialects]

        has_correction = all("corrected" in dialect_results[d] for d in dialects)

        if has_correction:
            corrected_wer = [dialect_results[d]["corrected"]["wer"] for d in dialects]
            corrected_cer = [dialect_results[d]["corrected"]["cer"] for d in dialects]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        x = np.arange(len(dialects))
        width = 0.35

        ax1.bar(x - width/2, raw_wer, width, label='Raw')
        if has_correction:
            ax1.bar(x + width/2, corrected_wer, width, label='Corrected')

        ax1.set_ylabel('Word Error Rate (WER)')
        ax1.set_title('WER by Dialect')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dialects)
        ax1.legend()

        ax2.bar(x - width/2, raw_cer, width, label='Raw')
        if has_correction:
            ax2.bar(x + width/2, corrected_cer, width, label='Corrected')

        ax2.set_ylabel('Character Error Rate (CER)')
        ax2.set_title('CER by Dialect')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dialects)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('./results/dialect_comparison.png')
        print("Dialect comparison plot saved to ./results/dialect_comparison.png")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ASR model")
    parser.add_argument("--model_path", type=str, default="./models/whisper-fine-tuned-final")
    parser.add_argument("--dataset_path", type=str, default="./data/processed_common_voice")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dialect", type=str, default=None)
    parser.add_argument("--no_correction", action="store_true")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="./results/evaluation_results.json")
    parser.add_argument("--by_dialect", action="store_true")

    args = parser.parse_args()

    os.makedirs("./results", exist_ok=True)

    evaluator = ASRModelEvaluator(model_path=args.model_path)

    if args.by_dialect:
        evaluator.evaluate_by_dialect(
            args.dataset_path,
            split=args.split,
            apply_correction=not args.no_correction,
            max_samples_per_dialect=args.max_samples
        )
    else:
        evaluator.evaluate_dataset(
            args.dataset_path,
            split=args.split,
            apply_correction=not args.no_correction,
            max_samples=args.max_samples,
            dialect=args.dialect,
            output_file=args.output_file
        )

if __name__ == "__main__":
    main()