# transcribe_all.py
import os
import sys
import json
import torch
import torchaudio
import glob
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio(audio_path, processor, model, device):
    """
    Transcribe audio using Whisper model
    
    Args:
        audio_path: Path to audio file
        processor: Whisper processor
        model: Whisper model
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
        
        # Process with Whisper
        input_features = processor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode the predicted IDs
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return f"Error: {str(e)}"

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe all audio files in a directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing audio files")
    parser.add_argument("--output_file", type=str, default="./data/transcriptions.json",
                        help="Output file to save transcriptions")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny",
                        help="Name of the Whisper model to use")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(args.data_dir, "*.mp3"))
    
    if not audio_files:
        print(f"No audio files found in {args.data_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Limit the number of files if specified
    if args.max_files and len(audio_files) > args.max_files:
        print(f"Limiting to {args.max_files} files")
        audio_files = audio_files[:args.max_files]
    
    # Transcribe audio files
    transcriptions = {}
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        file_id = os.path.basename(audio_file).replace("common_voice_en_", "").replace(".mp3", "")
        transcription = transcribe_audio(audio_file, processor, model, device)
        transcriptions[file_id] = {
            "path": audio_file,
            "transcription": transcription
        }
    
    # Save transcriptions
    print(f"Saving transcriptions to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(transcriptions, f, indent=2)
    
    print(f"Transcribed {len(transcriptions)} files")

if __name__ == "__main__":
    main()
