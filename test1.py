import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import numpy as np
import torch
import base64

# Set up HuggingFace models with language and temperature parameters
def setup_models(language, temperature):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model for speech-to-text
    transcriber = pipeline("automatic-speech-recognition", model=f"openai/whisper-{language}", device=0 if torch.cuda.is_available() else -1)
    
    # Load TinyLlama for text generation
    llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=0 if torch.cuda.is_available() else -1)
    
    # Load TTS model manually
    tts_model_name = "parler-tts/parler-tts-mini-v1"
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
    
    try:
        # Load TTS pipeline and ensure it runs on GPU
        tts_pipeline = pipeline("text-to-speech", model=tts_model_name, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.write(f"Warning: Could not load TTS pipeline. Error: {e}")
        tts_pipeline = None

    return transcriber, llm, tts_model, tokenizer, device, tts_pipeline

# Transcribe audio to text
def transcribe_audio(transcriber, audio_path):
    try:
        text = transcriber(audio_path)["text"]
        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Generate AI response with temperature control
def fetch_ai_response(llm, input_text, temperature):
    try:
        response = llm(
            input_text, 
            max_length=150,  
            do_sample=True, 
            top_p=0.85,
            temperature=temperature  # Use the temperature parameter
        )[0]["generated_text"]
        return response
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return None

# Convert text to audio
def text_to_audio(text, description, audio_path, tts_model, tokenizer, device, tts_pipeline):
    try:
        if tts_pipeline:
            tts_output = tts_pipeline(text)
            audio_arr = np.array(tts_output['audio']).astype(np.float32)
            sample_rate = tts_output['sampling_rate']
            
            st.write(f"Generated audio length: {len(audio_arr) / sample_rate:.2f} seconds")
            st.write(f"Sample rate: {sample_rate}")
        else:
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                output = tts_model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    max_length=512,
                    do_sample=True,
                    top_p=0.95
                )
                
            audio_arr = output.cpu().numpy().squeeze()
            sample_rate = 22050
            
        if len(audio_arr) == 0:
            st.error("Generated audio is empty.")
            return
        
        sf.write(audio_path, audio_arr, sample_rate)
        
    except Exception as e:
        st.error(f"Error converting text to audio: {e}")

# Display AI response in a styled card
def display_ai_response(response_text):
    st.markdown(
        f"""
        <div style="
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="
                margin-top: 0;
                color: #333;
            ">AI Response:</h4>
            <p style="
                margin: 0;
                color: #555;
            ">{response_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Autoplay audio file in browser
def auto_play_audio(audio_file):
    try:
        with open(audio_file, "rb") as audio_file_data:
            audio_bytes = audio_file_data.read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f'<audio src="data:audio/wav;base64,{base64_audio}" controls autoplay></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

# Main application
def main():
    st.sidebar.title("Customer Success Voicebot")
    st.title("Hi! How can I assist you today?")

    # Sidebar for parameter tuning
    language = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "de"])
    temperature = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Load models with selected language and temperature
    transcriber, llm, tts_model, tokenizer, device, tts_pipeline = setup_models(language, temperature)

    # Record audio from user
    recorded_audio = audio_recorder()

    if recorded_audio:
        audio_file = "audio.wav"
        with open(audio_file, "wb") as f:
            f.write(recorded_audio)

        # Transcribe the audio
        transcribed_text = transcribe_audio(transcriber, audio_file)
        if transcribed_text:
            st.write("Transcribed Text: ", transcribed_text)

            # Get AI response
            ai_response = fetch_ai_response(llm, transcribed_text, temperature)
            if ai_response:
                # Display AI response
                display_ai_response(ai_response)

                # Convert AI response to audio
                response_audio_file = "audio_response.wav"
                description = "A female speaker delivers a slightly expressive and animated speech."
                text_to_audio(ai_response, description, response_audio_file, tts_model, tokenizer, device, tts_pipeline)

                # Play the generated audio
                auto_play_audio(response_audio_file)

if __name__ == "__main__":
    main()
