import os, time
from dotenv import load_dotenv

load_dotenv()  #

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import glob
import subprocess
import math
from pydub import AudioSegment
import openai


from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.storage.file_system import LocalFileStore


# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1,
)



# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────
file_dir = os.path.dirname(os.path.realpath(__file__))
upload_dir = os.path.join(file_dir, '.cache/chunks')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

has_transcript = os.path.exists(os.path.join(file_dir, r'.cache/podcast.txt'))

@st.cache_resource()
def extract_audio_from_video(video_path):  
    if has_transcript: return  # 🐕‍🦺학습용  
    audio_path = video_path.replace("mp4", "mp3")  
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        audio_path,
        "-y", 
        ]
    subprocess.run(command)    


@st.cache_resource()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript: return  # 🐕‍🦺학습용  
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]

        exp_path = os.path.join(chunks_folder, f"chunk_{i}.mp3")
        chunk.export(exp_path, format="mp3")

@st.cache_resource()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript: return  # 🐕‍🦺학습용  
    files = glob.glob(os.path.join(chunk_folder, "chunk*.mp3"))
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:# append mode
            print(file, '녹취록 가져오는중...', end='')
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
           
            text_file.write(transcript.text) # 곧바로 텍스트 파일에 저장



# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🎤",
)
st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        label="Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = os.path.join(file_dir, rf".cache/{video.name}")
        audio_path = video_path.replace("mp4", "mp3")
        with open(video_path, 'wb') as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        chunks_folder = os.path.join(file_dir, r'./.cache/chunks')
        cut_audio_in_chunks(audio_path, 10, chunks_folder)

        status.update(label="Transcripting Audio...")
        transcript_path = video_path.replace("mp4", "txt")
        transcribe_chunks(chunks_folder, transcript_path)

    # 3개의 tab
    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    # transcript_tab
    with transcript_tab:
        with open(transcript_path, 'r') as file:
            st.write(file.read())

    # summary_tab
    with summary_tab:
        start = st.button("Generate Summary") 

        # ↓ 2개의 chain 을 시작
        #  첫번째 chain : 첫번째 document 를 요약 (summarize)
        #  두번째 chain : 다른 모든 document 를 요약
        #      LLM 에게 '이전의 summary' 와 새 context 를 사용하여 새로운 summary 를 만들게 함 (refine!).
        if start:
            loader = TextLoader(transcript_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)
            
            # 첫번째 chain : 첫번째 Document 요약
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            # 확인용
            summary = first_summary_chain.invoke({"text": docs[0].page_content})   # <- 첫번째 Document 입력

            st.write(summary) # 확인

            # TODO : 두번째 chain