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


# session_state 는 여러번 재실행해도 data 가 보존될수 있도록 해준다.
#   보존되는 데이터는 key-value 형태로 session에 저장됨

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

message = st.chat_input(placeholder="Send a message to AI")

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state['messages'].append({'message': message, 'role': role})
    
for msg in st.session_state['messages']:
    send_message(msg['message'], msg['role'], save=False)

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f'You said: {message}', 'ai')  

    with st.sidebar:
        st.write(st.session_state) # 확인용



# <확인>
# message 를 입력하면,  '추가' 되는것이 아니라, update 가 된다..

# 사용자가 무엇을 입력해도 비워지지 않고 남아있어야 한다!  어케 하나?
# 코드가 다시 실행되더라도 말이다.

# refresh 되더라도 상태값을 기억하도록
# streamlit 에서는 session state 제공.

# session state 는 여러번 재실행해도 data 가 보존될수 있도록 해준다.













