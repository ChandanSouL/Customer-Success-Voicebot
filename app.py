import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import base64


#intialize tinyllma client
def setup_openai_client(api_key):
    return openai.OpenAI(api_key=api_key)

#function to transcribe audio to text
def transcribe_audio(client,audio_path):
    
    with open(audio_path,"rb") as audio_file:
        transcribe = client.audio.transcriptions.create(model ="whisper-1",file = audio_file)
        return transcribe.text
#function to taking response from tinyllma
def fetch_ai_response(client, input_text):
    message = [{"role": "user", "context": input_text}]
    response = client.chat.completions.create(model="gpt-4o-mini",messages = message)
    return response.choices[0].message.content


#convert text to audio
def text_to_audio(client,text,audio_path):
    response = client.audio.speech.create(model = "tts-1",voice="echo",input=text)
    response.stream_to_file[audio_path]

def main():

    st.sidebar.title("API KEY CONFIGURATION")
    api_key = st.sidebar.text_input("Enter your API Key", type= "password")

    st.title("Customer Success Voicebot")
    st.write("Hi Welcome, How can I assist you today")

    if api_key:
        client = setup_openai_client(api_key)
        recorded_audio = audio_recorder()

        if recorded_audio:
            audio_file = "audio.mp3"
            with open(audio_file,"wb") as f:
                f.write(recorded_audio)
            
            transcribe_text = transcribe_audio(client,audio_file)
            st.write("Transscribed Text: ", transcribe_text)

            ai_response = fetch_ai_response(client,transcribe_text)
            response_audio_file = "audio_response.mp3"
            text_to_audio(client,ai_response,response_audio_file)
            st.audio(response_audio_file)

if __name__ == "__main__":
    main()
