import os
import whisper
import streamlit as st
from pydub import AudioSegment
import pandas as pd

st.set_page_config(
    page_title="Whisper based ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

audio_tags = {'comments': 'Converted using pydub!'}

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"

fraud_keywords = [
    'Class after token payment',
    'Upfront Payment',
    'Bootcamp',
    'Token amount is refundable',
    'Get server after 50 percent of total fee',
    'Job guarantee',
    '100% placement guarantee',
    'Personal account',
    'Refund',
    'S4 Hana',
    'Server Access',
    'Free classes',
    'Free',
    'Free Days',
    'Free trial',
    'Trial classes',
    'My account',
    'First month free',
    'Free services',
    'cancellation policy',
    'Cancel'
]

def split_audio(input_audio_path, chunk_duration_ms):
    audio = AudioSegment.from_file(input_audio_path)
    duration = len(audio)
    chunks = []
    for i in range(0, duration, chunk_duration_ms):
        start = i
        end = min(i + chunk_duration_ms, duration)
        chunk = audio[start:end]
        chunks.append(chunk)
    return chunks

def to_mp3(audio_data, output_audio_file):
    audio_data.export(os.path.join(download_path, output_audio_file), format="mp3", tags=audio_tags)
    return output_audio_file

def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file), "w") as f:
        f.write(transcript_data)

st.title("Automatic Speech Recognition")
st.info('Supports Audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV')
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"])

audio_file = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        chunks = split_audio(os.path.join(upload_path, uploaded_file.name), chunk_duration_ms=60000) # Splitting into chunks of 1 minute
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = to_mp3(chunk, f"{i}_{output_audio_file}")
            chunk_files.append(os.path.join(download_path, chunk_file))
        audio_files = ' '.join(chunk_files)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Feel free to play your uploaded audio file")
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    if st.button("Generate Transcript"):
        with st.spinner(f"Generating Transcript"):
            transcript = ""
            for chunk_file in chunk_files:
                chunk_transcript = process_audio(chunk_file, whisper_model_type.lower())
                transcript += chunk_transcript + "\n"
            output_txt_file = uploaded_file.name.split('.')[0] + ".txt"
            save_transcript(transcript, output_txt_file)
            output_file = open(os.path.join(transcript_path, output_txt_file), "r")
            output_file_data = output_file.read()

            # Fraud detection
            detected_keywords = [keyword for keyword in fraud_keywords if keyword.lower() in output_file_data.lower()]
            fraud_detected = len(detected_keywords) > 0

            output_df = pd.DataFrame({
                'Uploaded File Name': [uploaded_file.name],
                'Output File Data': [output_file_data],
                'Detected Keywords': [detected_keywords],
                'Fraud Detected': [fraud_detected]
            })

        if st.download_button(
                label="Download Transcript",
                data=output_file_data,
                file_name=output_txt_file,
                mime='text/plain'
        ):
            st.balloons()
            st.success(' Download Successful ')

else:
    st.warning(' Please upload your audio file ')

# Display the result dataframe
if 'output_df' in locals():
    st.subheader("Fraud Detection Result:")
    st.write(output_df)

st.markdown(
    "<br><hr><center>Made by <a href='mailto:rigved.sarougi@henryharvin.com?subject=ASR Whisper WebApp!&body=Please specify the issue you are facing with the app.'><strong>Rigved Sarougi</strong><hr>",
    unsafe_allow_html=True)
