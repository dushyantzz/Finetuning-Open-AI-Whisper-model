# transcribe_audio.py
import os
import argparse
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Dataset
from tqdm import tqdm

def transcribe_audio(audio_path, model, processor, device):
    """
    Transcribe a single audio file using Whisper
    
    Args:
        audio_path: Path to the audio file
        model: Whisper model
        processor: Whisper processor
        device: Device to run the model on
        
    Returns:
        Transcription text
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Process audio with Whisper
        input_features = processor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        
        # Decode the generated IDs
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return f"Error: {str(e)}"

def main():
    """Transcribe audio files in the dataset"""
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("--dataset_path", type=str, default="./data/processed_common_voice",
                        help="Path to the processed dataset")
    parser.add_argument("--output_path", type=str, default="./data/transcribed_dataset",
                        help="Path to save the transcribed dataset")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny",
                        help="Name or path of the Whisper model")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Whisper model
    print(f"Loading Whisper model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process each split
    for split in ["train", "validation", "test"]:
        try:
            # Load dataset
            dataset_path = os.path.join(args.dataset_path, split)
            if not os.path.exists(dataset_path):
                print(f"Split {split} not found at {dataset_path}")
                continue
            
            dataset = load_from_disk(dataset_path)
            print(f"Loaded {len(dataset)} examples from {dataset_path}")
            
            # Transcribe audio files
            transcriptions = []
            for example in tqdm(dataset, desc=f"Transcribing {split} split"):
                audio_path = example["path"]
                transcription = transcribe_audio(audio_path, model, processor, device)
                
                # Create a new example with the transcription
                new_example = example.copy()
                new_example["sentence"] = transcription
                transcriptions.append(new_example)
            
            # Create a new dataset with the transcriptions
            transcribed_dataset = Dataset.from_list(transcriptions)
            
            # Save the transcribed dataset
            output_path = os.path.join(args.output_path, split)
            print(f"Saving {len(transcribed_dataset)} examples to {output_path}")
            transcribed_dataset.save_to_disk(output_path)
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
    
    print("Transcription completed!")

if __name__ == "__main__":
    main()
