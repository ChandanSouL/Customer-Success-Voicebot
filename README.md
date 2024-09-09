**Steps to Configure and Run**

Open Google Colab/ Kaggale Notebook
1. Select Nvidia-T4 GPU
2. Go to Notebook settings and select T4 as Hardware Accelator

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

```!streamlit run /content/Customer-Success-Voicebot/mainapp.py & npx localtunnel --port 8501```

You will get an external link from the above command open it in a new tab and use the Public Ip from the above command to login into the Localtunnel website to view streamlit Application.

**Specific Libraries <Debug>**

**Protobuf Required Library**

```!pip install protobuf==3.20.*```

**Base64 import error**

```!pip install base64```


