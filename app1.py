import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import numpy as np
import torch

# Set up HuggingFace models
def setup_models():
    # Load Whisper model for speech-to-text
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

    # Load TinyLlama for text generation
    llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load TTS model manually from Hugging Face
    tts_model_name = "parler-tts/parler-tts-mini-v1"
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tts_model_name)

    # Check if TTS pipeline is available; if not, use manual TTS methods
    try:
        tts_pipeline = pipeline("text-to-speech", model=tts_model_name)
    except:
        tts_pipeline = None

    return transcriber, llm, tts_model, tokenizer, device,tts_pipeline

# Transcribe audio to text
def transcribe_audio(transcriber, audio_path):
    text = transcriber(audio_path)["text"]
    return text

# Generate AI response
def fetch_ai_response(llm, input_text):
    print(f"Input Text: {input_text}")
    response = llm(input_text, max_length=100, do_sample=True, top_p=0.9)[0]["generated_text"]
    print(f"AI Response: {response}")
    return response

# Convert text to speech
def text_to_audio(text, description, audio_path, tts_model, tokenizer, device,tts_pipeline):
    # Prepare input for TTS model
    if tts_pipeline:
        # Use TTS pipeline
        tts_output = tts_pipeline(text)
        audio_arr = np.array(tts_output['audio']).astype(np.float32)
        sample_rate = tts_output['sampling_rate']
    else:
        # Fallback manual method if pipeline is not available
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = tts_model.generate(input_ids=input_ids, max_length=512, do_sample=True, top_p=0.95)

        audio_arr = output.cpu().numpy().squeeze()
        sample_rate = 22050  # Default TTS sample rate

    # Save audio file
    sf.write(audio_path, audio_arr, sample_rate)
    print(f"Audio saved to: {audio_path}")

# Main application
def main():
    st.sidebar.title("Customer Success Voicebot")
    st.title("Hi! How can I assist you today?")

    # Load models
    transcriber, llm, tts_model, tokenizer, device, tts_pipeline = setup_models()

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
        description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        text_to_audio(ai_response, description, response_audio_file, tts_model, tokenizer, device,tts_pipeline)
        st.audio(response_audio_file)

if __name__ == "__main__":
    main()
