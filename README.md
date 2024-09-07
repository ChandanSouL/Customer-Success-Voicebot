!pip install sounddevice scipy transformers datasets accelerate soundfile torchaudio streamlit audio_recorder_streamlit
!pip install git+https://github.com/openai/whisper.git
!apt-get install libportaudio2
!pip install git+https://github.com/huggingface/parler-tts.git
!wget -q -O - ipv4.icanhazip.com
!streamlit run /content/Customer-Success-Voicebot/maintest.py & npx localtunnel --port 8501


####additional
!pip install --upgrade transformers
!pip install protobuf==3.20.*