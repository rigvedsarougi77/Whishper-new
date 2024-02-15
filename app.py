import streamlit as st
import whisper
st.set_page_config(page_title="chatPdf", page_icon="🧊")
st.title("Audio transciption app")

#upload audio file with streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav","mp3","mp4"])
#importing model -- base(74M pararameter)
model = whisper.load_model("base")
st.info("Whisper model loaded")

st.header("Play audio file:")
st.audio(audio_file)

if st.button("Transcribe Audio"):
    if audio_file is not None:
        st.success("Transcribing Audio")
        transcription = model.transcribe(audio_file.name)
        st.success("Transcription Complete")
        st.markdown(transcription["text"])
    else:
        st.error("Please upload a audio file")
