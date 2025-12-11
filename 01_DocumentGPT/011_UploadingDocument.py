import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.storage.file_system import LocalFileStore
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1,
)


# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────

# 업로드할 파일, 임베딩 벡터를 저장할 경로. 미리 생성해 두기
upload_dir = r'./.cache/files'
embedding_dir = r'./.cache/embeddings'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


def embed_file(file):
    # 업로드한 파일 저장
    file_content = file.read()
    file_path = os.path.join(upload_dir, file.name)
    with open(file_path, 'wb') as f:
        f.write(file_content)


    # 업로드된 파일을 load & split
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # embeddings 생성하기 + cache 하기
    # 업로드된 '각각의 파일' 별로 embedding cache 디렉토리 지정하여 준비
    cache_dir = LocalFileStore(os.path.join(embedding_dir, file.name))
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # 위 embedding 을 vectorstore 에 넣기
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # retriever 리턴하기
    retriever = vectorstore.as_retriever()
    return retriever




# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown("""
안녕하세요!
이 챗봇을 사용해서 여러분의 파일들에 대해 AI에게 물어보세요
""")

with st.sidebar:
    file = st.file_uploader(
        label="Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    # file_content = file.read()
    # st.write(file_content)
    retriever = embed_file(file)

    # 동작확인
    docs = retriever.invoke("ministry of truth")
    st.write(docs)