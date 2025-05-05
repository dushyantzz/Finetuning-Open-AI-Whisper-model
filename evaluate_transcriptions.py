import os
import json
import argparse
import matplotlib.pyplot as plt
from jiwer import wer, cer

def evaluate_transcriptions(reference_file, hypothesis_file, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    with open(reference_file, "r") as f:
        references = json.load(f)

    with open(hypothesis_file, "r") as f:
        hypotheses = json.load(f)

    results = {
        "file_id": [],
        "reference": [],
        "hypothesis": [],
        "wer": [],
        "cer": []
    }
    for file_id, ref_data in references.items():
        if file_id not in hypotheses:
            print(f"Warning: File ID {file_id} not found in hypothesis file")
            continue

        reference = ref_data["transcription"]
        hypothesis = hypotheses[file_id]["transcription"]

        file_wer = wer(reference, hypothesis)
        file_cer = cer(reference, hypothesis)

        results["file_id"].append(file_id)
        results["reference"].append(reference)
        results["hypothesis"].append(hypothesis)
        results["wer"].append(file_wer)
        results["cer"].append(file_cer)

    avg_wer = sum(results["wer"]) / len(results["wer"]) if results["wer"] else 0
    avg_cer = sum(results["cer"]) / len(results["cer"]) if results["cer"] else 0

    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "files": results
        }, f, indent=2)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results["file_id"])), results["wer"], label="WER")
    plt.bar(range(len(results["file_id"])), results["cer"], label="CER", alpha=0.7)
    plt.xlabel("File ID")
    plt.ylabel("Error Rate")
    plt.title(f"Transcription Error Rates (Avg WER: {avg_wer:.4f}, Avg CER: {avg_cer:.4f})")
    plt.xticks(range(len(results["file_id"])), results["file_id"], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_rates.png"))

    return {
        "avg_wer": avg_wer,
        "avg_cer": avg_cer
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription quality")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--hypothesis", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()

    metrics = evaluate_transcriptions(args.reference, args.hypothesis, args.output_dir)

    print(f"Average WER: {metrics['avg_wer']:.4f}")
    print(f"Average CER: {metrics['avg_cer']:.4f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
