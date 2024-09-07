import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline, AutoTokenizer, WhisperForConditionalGeneration
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import WhisperProcessor

# Set up HuggingFace models
def setup_models():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model for speech-to-text
    transcriber = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device)
    
    # Load TinyLlama for text generation
    llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=device)
    
    # Load TTS model manually from Hugging Face
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    return transcriber, llm, tts_model, tokenizer

# Transcribe audio to text
def transcribe_audio(transcriber, audio_path):
    # Use WhisperProcessor to preprocess the audio
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    
    # Read the audio file
    audio_input, _ = sf.read(audio_path)

    # Process the audio input
    audio_input = processor(audio_input, return_tensors="pt").input_features
    audio_input = audio_input.to(transcriber.device)
    
    # Generate transcription
    with torch.no_grad():
        logits = transcriber.generate(input_ids=audio_input)
    
    transcription = processor.decode(logits[0], skip_special_tokens=True)
    return transcription

# Generate AI response
def fetch_ai_response(llm, input_text):
    response = llm(input_text, max_length=100)[0]["generated_text"]
    return response

# Convert text to speech
def text_to_audio(tts_model, tokenizer, text, audio_path):
    # Tokenize and generate speech
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(tts_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = tts_model.generate(**inputs)
    
    # Convert model output to numpy array
    audio_data = outputs[0].cpu().numpy()
    
    # Save audio data to file
    sf.write(audio_path, audio_data, 22050)  # Make sure sample rate matches your model's requirements

# Main application
def main():
    st.sidebar.title("Customer Success Voicebot")
    st.title("Hi! How can I assist you today?")

    # Load models
    transcriber, llm, tts_model, tokenizer = setup_models()
    
    # Record audio from user
    recorded_audio = audio_recorder()

    if recorded_audio:
        audio_file = "audio.wav"
        with open(audio_file, "wb") as f:
            f.write(recorded_audio)

        # Transcribe the audio
        transcribed_text = transcribe_audio(transcriber, audio_file)
        st.write("Transcribed Text: ", transcribed_text)

        # Get AI response
        ai_response = fetch_ai_response(llm, transcribed_text)
        st.write("AI Response: ", ai_response)

        # Convert the AI response to audio
        response_audio_file = "audio_response.wav"
        text_to_audio(tts_model, tokenizer, ai_response, response_audio_file)
        st.audio(response_audio_file)

if __name__ == "__main__":
    main()
