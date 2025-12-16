import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_text_splitters.character import CharacterTextSplitter

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.retrievers.wikipedia import WikipediaRetriever

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────



# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────

file_dir = os.path.dirname(os.path.realpath(__file__)) # *.py 파일의 '경로'만
upload_dir = os.path.join(file_dir, '.cache/quiz_files')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# split_file()
# vector, embedding 필요없다. 오로지 문서가 필요하고,
# 그 문서들을 split 까지만 해두면 된다.
@st.cache_resource(show_spinner="Loading file...")  # ← 이 split_file() 함수를 embed 되진 않고 caching 만 될거다.
def split_file(file):  # ←함수명 변경
    file_content = file.read()
    file_path = os.path.join(upload_dir, file.name)
   
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    # ※ DocumentGPT 에 있었던 이하 embeddings 나 vectorstore 등을 필요없다.
    #     embed 하지 않을거고 어떤 검색도 하지 않을거다.
    #     단지 'text file' 을 넣어줄거고,  그 문서들로부터 quiz 를 만들거다.
    return docs  # split 한 List[Document] 리턴!


# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="QuizGPT",
    page_icon="👩‍🚒",
)

st.title("QuizGPT")

with st.sidebar:

    docs = None  # 읽어들인 문서들 List[Document]

    choice = st.selectbox(
        label="Choose what you want to use.",
        options=(
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],            
        )
        # 파일 업로드 처리 구현
        if file:
            docs = split_file(file)
            st.write(len(docs), '개의 Document 로 split')
            st.write(docs) # 확인용

    else:
        topic = st.text_input("Search Wikipedia...")
        # Wikipedia Retriever 사용
        if topic:
            # top_k_results=1 : retieve 결과중 첫번째 문서만!
            retriever = WikipediaRetriever(top_k_results=5)

            with st.status("Searching Wikipedia..."):
                docs = retriever.invoke(topic)
                st.write(len(docs), '개의 문서 retrieve') # 확인용
                st.write(docs)




























