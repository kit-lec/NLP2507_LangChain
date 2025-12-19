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
    return  {
        "question": question,
        "answers": [
            # 각각의 Document 마다 Dict 생성
            {
                "answer": answers_chain.invoke({
                            "question": question,
                            "context": doc.page_content,
                        }).content,
                "source": doc.metadata['source'], # 출처 url
                "date": doc.metadata['lastmod'], # 페이지의 마지막 수정날짜
            }
            
            for doc in docs
        ]
    }


choose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources as it is. Keep it as a link
     
            Answers: {answers}
    """),
    ('human', "{question}")
])


# 입력: '모든 answer' 와 '사용자 question'
# 출력: 선택된 '최종 answer'
def choose_answer(inputs):
    answers = inputs['answers']
    question = inputs['question']
    choose_chain = choose_prompt | llm

    # 압축할 string 을 저장할 변수
    condensed = ""
    for answer in answers:
        condensed += f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"

    # st.write(condensed)  # 확인용!

    return choose_chain.invoke({
        "question": question,
        
        # "answers": answers,   
        #        ↑ 잠깐! answers 는
        #     [{"answer":.., "source".., "date":...}] 형태로 받아왔다.
        #    이게 prompt 에 string 형식으로 전달되는 셈이다.
        #    ↓ 리팩터 해보자

        "answers": condensed,

    })

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
    loader.max_depth=3
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
        query = st.text_input("Ask a question to the website.")

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

        if query:
            chain = {
                    "docs": retriever, 
                    "question": RunnablePassthrough()
                } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
                        # ↑ get_answers의 출력값이 → choose_answer 의 입력값은 될거다
                        # choose_answer 는 두가지 가 필요하다
                        #    answer(답변들) 과 사용자의 question 이다.
                        #    이를 가지고 LLM 에게 요청할것 이기 때문이다
                        # 그렇게 하려면 get_answers 의 리턴값은 List 가 아니라 Dict 이어야 한다

            result = chain.invoke(query)
            st.markdown(result.content)












