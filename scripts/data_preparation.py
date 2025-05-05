from datasets import load_dataset, Audio, Dataset, concatenate_datasets
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import random

def prepare_common_voice_dataset(dialects=None, split_ratio=0.1, test_ratio=0.05, local_data_path=None):
    if local_data_path:
        dataset = load_local_common_voice(local_data_path)
    else:
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")

    if dialects:
        if isinstance(dialects, str):
            dialects = [dialects]

        filtered_datasets = []
        for dialect in dialects:
            dialect_dataset = dataset.filter(lambda example: example["accent"] == dialect)
            filtered_datasets.append(dialect_dataset)
            print(f"Found {len(dialect_dataset)} examples for dialect '{dialect}'")

        if filtered_datasets:
            dataset = concatenate_datasets(filtered_datasets)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    splits = dataset.train_test_split(test_size=split_ratio + test_ratio)
    train_dataset = splits['train']

    test_val_ratio = test_ratio / (split_ratio + test_ratio)
    test_val_splits = splits['test'].train_test_split(test_size=test_val_ratio)
    val_dataset = test_val_splits['train']
    test_dataset = test_val_splits['test']

    print(f"Dataset prepared with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test examples")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }

def load_local_common_voice(data_path):
    print(f"Loading data from {data_path}")
    audio_files = glob.glob(os.path.join(data_path, "*.mp3"))

    if not audio_files:
        raise ValueError(f"No mp3 files found in {data_path}")

    data = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        file_id = os.path.basename(audio_file).replace("common_voice_en_", "").replace(".mp3", "")

        data.append({
            "path": audio_file,
            "audio": {"path": audio_file},
            "sentence": f"Placeholder transcription for file {file_id}",  # Will be replaced with actual transcription
            "accent": "unknown",  # We'll try to detect this based on the audio
            "file_id": file_id
        })

    return Dataset.from_list(data)

def transcribe_dataset_with_whisper(dataset, batch_size=16, device=None):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Whisper model...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    def process_batch(examples):
        audio_arrays = []
        for audio_path in examples["path"]:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

                audio_arrays.append(waveform.squeeze().numpy())
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                audio_arrays.append(np.zeros(16000))

        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features)

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        examples["sentence"] = transcriptions

        return examples

    print("Transcribing audio files...")
    transcribed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        desc="Transcribing with Whisper"
    )

    return transcribed_dataset

def apply_data_augmentation(dataset, augmentation_factor=0.3):
    def augment_audio(example):
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]

        aug_type = random.choice(["noise", "none"])

        if aug_type == "noise":
            noise_level = random.uniform(0.005, 0.015)
            noise = torch.randn_like(torch.tensor(audio)) * noise_level
            augmented = torch.tensor(audio) + noise

        else:
            augmented = torch.tensor(audio)

        example["audio"]["array"] = augmented.numpy()
        example["augmentation"] = aug_type

        return example

    num_to_augment = int(len(dataset) * augmentation_factor)
    indices_to_augment = random.sample(range(len(dataset)), num_to_augment)

    subset_to_augment = dataset.select(indices_to_augment)
    augmented_subset = subset_to_augment.map(augment_audio)

    combined_dataset = concatenate_datasets([dataset, augmented_subset])

    print(f"Added {len(augmented_subset)} augmented examples to dataset")
    return combined_dataset

def prepare_dataset_for_training(dataset_dict, augment=True):
    if augment:
        dataset_dict['train'] = apply_data_augmentation(dataset_dict['train'])

    return dataset_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for ASR training")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/processed_common_voice")
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--no_augmentation", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Loading data from {args.data_path}")

        raw_dataset = load_local_common_voice(args.data_path)

        if args.max_files and len(raw_dataset) > args.max_files:
            print(f"Limiting to {args.max_files} files")
            raw_dataset = raw_dataset.select(range(args.max_files))

        if args.transcribe:
            print("Transcribing audio files with Whisper...")
            raw_dataset = transcribe_dataset_with_whisper(
                raw_dataset,
                batch_size=args.batch_size
            )

        train_size = int(0.8 * len(raw_dataset))
        val_size = int(0.1 * len(raw_dataset))
        test_size = len(raw_dataset) - train_size - val_size

        splits = raw_dataset.train_test_split(test_size=val_size + test_size)
        train_dataset = splits['train']

        test_val_ratio = test_size / (val_size + test_size)
        test_val_splits = splits['test'].train_test_split(test_size=test_val_ratio)
        val_dataset = test_val_splits['train']
        test_dataset = test_val_splits['test']

        dataset_dict = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

        if not args.no_augmentation:
            print("Applying data augmentation...")
            dataset_dict['train'] = apply_data_augmentation(dataset_dict['train'])

        for split, dataset in dataset_dict.items():
            output_path = os.path.join(args.output_dir, split)
            print(f"Saving {split} split with {len(dataset)} examples to {output_path}")
            dataset.save_to_disk(output_path)

        print(f"Processed dataset saved to {args.output_dir}")

        print("\nSample from training set:")
        print(dataset_dict['train'][0])

    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

        print("Creating a small example dataset for testing...")

        example_data = []
        for i in range(10):
            example_data.append({
                "path": f"example_{i}.mp3",
                "audio": {"path": f"example_{i}.mp3", "array": np.zeros(16000), "sampling_rate": 16000},
                "sentence": f"This is example sentence {i}",
                "accent": "unknown",
                "file_id": f"example_{i}"
            })

        example_dataset = Dataset.from_list(example_data)

        train_size = int(0.8 * len(example_dataset))
        val_size = int(0.1 * len(example_dataset))
        test_size = len(example_dataset) - train_size - val_size

        splits = example_dataset.train_test_split(test_size=val_size + test_size)
        train_dataset = splits['train']

        test_val_ratio = test_size / (val_size + test_size)
        test_val_splits = splits['test'].train_test_split(test_size=test_val_ratio)
        val_dataset = test_val_splits['train']
        test_dataset = test_val_splits['test']

        dataset_dict = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

        for split, dataset in dataset_dict.items():
            output_path = os.path.join(args.output_dir, split)
            print(f"Saving {split} split with {len(dataset)} examples to {output_path}")
            dataset.save_to_disk(output_path)

        print(f"Example dataset saved to {args.output_dir}")
