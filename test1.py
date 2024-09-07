import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Set up HuggingFace models
def setup_models():
    # Load Whisper model for speech-to-text
    transcriber = pipeline(model="openai/whisper-tiny.en", task="automatic-speech-recognition")
    
    # Load TinyLlama for text generation
    llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Load TTS model manually from Hugging Face
    tts_model = "parler-tts/parler-tts-mini-v1"
    
    return transcriber, llm, tts_model

# Transcribe audio to text
def transcribe_audio(transcriber, audio_path):
    text = transcriber(audio_path)["text"]
    return text

# Generate AI response
def fetch_ai_response(llm, input_text):
    response = llm(input_text, max_length=100)[0]["generated_text"]
    return response

# Convert text to speech
def text_to_audio(tts_model, text, audio_path):
    # Download the TTS model file
    model = hf_hub_download(repo_id=tts_model, filename="pytorch_model.bin")
    tokenizer = hf_hub_download(repo_id=tts_model, filename="tokenizer.json")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    # Tokenize and generate speech
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    audio_data = outputs[0].numpy()  # Convert model output to numpy array
    sf.write(audio_path, audio_data, 16000)

# Main application
def main():
    st.sidebar.title("Customer Success Voicebot")
    st.title("Hi! How can I assist you today?")

    # Load models
    transcriber, llm, tts_model = setup_models()
    
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
        text_to_audio(tts_model, ai_response, response_audio_file)
        st.audio(response_audio_file)

if __name__ == "__main__":
    main()
