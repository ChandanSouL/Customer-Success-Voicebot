import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import numpy as np
import torch
import base64

# Set up HuggingFace models
def setup_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model for speech-to-text (ensuring it runs on GPU)
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=0 if torch.cuda.is_available() else -1)
    
    # Load TinyLlama for text generation (ensuring it runs on GPU)
    llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=0 if torch.cuda.is_available() else -1)
    
    # Load TTS model manually
    tts_model_name = "parler-tts/parler-tts-mini-v1"
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
    
    try:
        # Load TTS pipeline and ensure it runs on GPU
        tts_pipeline = pipeline("text-to-speech", model=tts_model_name, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        #st.write(f"Warning: Could not load TTS pipeline. Error: {e}")
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

# Generate AI response
def fetch_ai_response(llm, input_text):
    try:
        response = llm(
            input_text, 
            max_length=50,  # Increased max length for more detailed responses
            do_sample=True, 
            top_p=0.85  # Adjusting top_p for more control over randomness
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
            
            # Debugging information
            #st.write(f"Generated audio length: {len(audio_arr) / sample_rate:.2f} seconds")
            #st.write(f"Sample rate: {sample_rate}")
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
                
            # Convert generated tokens to audio
            audio_arr = output.cpu().numpy().squeeze()
            sample_rate = 22050  # Default TTS sample rate
            
            # Debugging information
            #st.write(f"Generated audio length: {len(audio_arr) / sample_rate:.2f} seconds")
            #st.write(f"Sample rate: {sample_rate}")

        # Ensure audio array is not empty
        if len(audio_arr) == 0:
            st.error("Generated audio is empty.")
            return
        
        # Save audio file
        sf.write(audio_path, audio_arr, tts_model.config.sampling_rate)
        #st.write(f"Audio saved to: {audio_path}")

        # Provide a download link for the audio file
        #st.audio(audio_path)
        #st.download_button(label="Download Audio", data=open(audio_path, "rb"), file_name="response_audio.wav")

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
        if transcribed_text:
            st.write("Transcribed Text: ", transcribed_text)

            # Get AI response
            ai_response = fetch_ai_response(llm, transcribed_text)
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
