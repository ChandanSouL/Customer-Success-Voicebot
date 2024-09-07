Steps to Configure and Run

**Clone this project in your Google collab notebook/ Kaggale Notebook**

```git clone https://github.com/ChandanSouL/Customer-Success-Voicebot.git```

**Install the required libraries**

```!pip install -r requirements.txt```

**Install Whisper Pre-Trained model from the github**

```!pip install git+https://github.com/openai/whisper.git```

**Install Parler-TTS Pre-trained model from the github**

```!pip install git+https://github.com/huggingface/parler-tts.git```

**Run a shell command to fetch your external IP address using wget from the terminal**

```!wget -q -O - ipv4.icanhazip.com```

**Run Streamlit to Run your app on localtunnel use the IP from the above and login**

```!streamlit run /content/Customer-Success-Voicebot/maintest.py & npx localtunnel --port 8501```
