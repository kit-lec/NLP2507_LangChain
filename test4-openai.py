import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import ChatOpenAI
chat = ChatOpenAI()  # 반드시 OPENAI_API_KEY 환경변수가 설정되어 있어야 한다
                     # .env 를 읽는다.

result = chat.invoke("How many planets are there?")
print(result)