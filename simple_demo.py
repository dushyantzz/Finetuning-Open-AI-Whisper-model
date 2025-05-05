# simple_demo.py
import os
import torch
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Global variables for model and processor
model = None
processor = None

def load_model(model_name="openai/whisper-tiny"):
    """
    Load Whisper model and processor

    Args:
        model_name: Name of the Whisper model to use

    Returns:
        Tuple of (processor, model)
    """
    global model, processor

    if model is None or processor is None:
        print(f"Loading model: {model_name}")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model.to(device)

    return processor, model

def transcribe(audio_file, model_name="openai/whisper-tiny"):
    """
    Transcribe audio using Whisper model

    Args:
        audio_file: Path to audio file
        model_name: Name of the Whisper model to use

    Returns:
        Transcription text
    """
    # Load model and processor
    processor, model = load_model(model_name)

    # Get device
    device = next(model.parameters()).device

    # Load and process audio
    import torchaudio

    try:
        # Load audio
        audio_array, sampling_rate = torchaudio.load(audio_file)

        # Resample if needed
        if sampling_rate != 16000:
            audio_array = torchaudio.functional.resample(audio_array, sampling_rate, 16000)

        # Convert to mono if needed
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)

        # Process with Whisper
        input_features = processor(
            audio_array.squeeze().numpy(),
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
        return f"Error: {str(e)}"

def transcribe_with_model_selection(audio_file, model_name):
    """
    Transcribe audio with model selection

    Args:
        audio_file: Path to audio file
        model_name: Name of the Whisper model to use

    Returns:
        Transcription text
    """
    return transcribe(audio_file, model_name)

# Create Gradio interface
demo = gr.Interface(
    fn=transcribe_with_model_selection,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Dropdown(
            choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
            value="openai/whisper-tiny",
            label="Model"
        )
    ],
    outputs=gr.Textbox(label="Transcription"),
    title="Speech Recognition Demo",
    description="Upload audio to transcribe with Whisper model. You can select different model sizes for better accuracy (at the cost of speed)."
)

if __name__ == "__main__":
    # Load model at startup
    load_model()

    # Launch demo
    demo.launch()
