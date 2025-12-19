import os, time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate

from langchain_community.document_loaders.sitemap import SitemapLoader

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                 
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                 
    Examples:
                                                 
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                 
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                 
    Your turn!

    Question: {question}
""")


def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    # retriever 가 건네준 document 들을 각가 처리할 chain 을 만들어 보자.
    answers_chain = answers_prompt | llm
    answers = [
        answers_chain.invoke({
            "question": question,
            "context": doc.page_content,
        }).content
        for doc in docs
    ]

    st.write(answers)  # 결과 확인용.

# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────

def parse_page(soup):
    
    header = soup.select_one("#header")
    footer = soup.select_one("#footer")

    if header:
        header.decompose()
    
    if footer:
        footer.decompose()

    return ( 
        str(soup.get_text())
        .replace("\\n", " ")
        .replace("\xa0", " ")
        .replace("Filter by category", "")
    )


@st.cache_resource(show_spinner="Fetching URL...")
def load_website(url):

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[
        #     r"^(.*\/news\/).*",
        # ],
        parsing_function=parse_page,
    )
    loader.max_depth=1
    # loader.requests_per_second = 3

    loader.headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36'}
    docs = loader.load_and_split(text_splitter=splitter)

    vector_store = FAISS.from_documents(
        documents=docs,
        # ★명심. cache 를 만들때..
        #   다른 sitemap 에서 얻은 각각의 URL 마다 별도의 cache를 만들어야 한다
        embedding=OpenAIEmbeddings(),
    )

    return vector_store.as_retriever()

# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
"""
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:

    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)

        # Map Re-Rank Chain 만들기. 두개의 chain 이 필요하다
        # 1.첫번째 chain
        #   모든 개별 Document 에 대한 답변 생성 및 채점 담당
        # 2.두번째 chain
        #   모든 답변을 가진 마지막 시점에 실행된다
        #   점수가 제일 높고 + 가장 최신 정보를 담고 있는 답변들 고른다
    
        # ----------
        # 🟡 첫번째 chain
        #    retreiver 에 의해 리턴된 List[Document] 와 사용자가 입력한 question 필요
        #    이는 chain 의 입력값들이다.
        chain = {
                "docs": retriever, 
                "question": RunnablePassthrough()
            } | RunnableLambda(get_answers)

        chain.invoke("In what year was Mistral AI established?")












