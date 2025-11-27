import streamlit as st
import os
import time

import numpy as np

from dotenv import load_dotenv
load_dotenv()


print(f'✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...')


st.set_page_config (
    page_title = "Chat Messages",
    page_icon = "😎",
)

st.title('Chat Messages')

# chat_message()  : chat message container 생성
#             human 혹은 AI 모두에게서 받을수 있다.
#     매개변수는 'user', 'assistant', 'ai', 'human' 중 하나

with st.chat_message(name='human'):
    st.write('helllo')
    st.write('how are you?')

with st.chat_message(name='ai'):
    st.write('helllo')
    st.write('how are you?')

with st.chat_message(name='user'):
    st.write('helllo')
    st.write('how are you?')

with st.chat_message(name='assistant'):
    st.write('helllo')
    st.write('how are you?')














