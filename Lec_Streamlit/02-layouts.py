import streamlit as st
import os
import time

import numpy as np

from dotenv import load_dotenv
load_dotenv()


print(f'✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...')

st.title('layout')

# layout
#  streamlit 에서 제공하는 다양한 레이아웃
#  공식: https://docs.streamlit.io/develop/api-reference/layout  (◀ 함 보자!)

# ────────────────────────────────────────────────────────
# container vs. empty

# 레이아웃 사용방식
# 방식1
cont = st.container(border=True)
cont.write('container 내부의 요소')
cont.markdown('container **내부** 의 요소')

st.write("container 바깥의 요소")

cont.write('이건 어디 그려질까요?')


# 방식2 (추천) with 사용
with st.container(border=True):
    st.write('컨테이너 안의 요소')
    st.bar_chart(np.random.randn(50, 3))

# container() : 여러 요소들을 담는다
# empty() : 한개의 요소만 담는다.

with st.empty():
    st.write("강아지")
    st.write("고양이")

st.title("sidebar")

with st.sidebar:
    st.title('sidebar title')
    st.text_input("이름을 입력하세요")
    "Hello everyone"


st.title("tabs")
tab_one, tab_two, tab_three = st.tabs(["하나", "둘", "셋"])

with tab_one:
    st.subheader('alpha')

with tab_two:
    st.subheader('bravo')

with tab_three:
    st.subheader('charlie')

st.title("columns")
col1, col2, col3 = st.columns(3)    
with col1:
    st.metric(label="달러USD", value="1,228 원", delta="-12.00 원")
with col2:
    st.metric(label="일본JPY(100엔)", value="958.63 원", delta="-7.44 원")
with col3:
    st.metric(label="유럽연합EUR", value="1,335.82 원", delta="11.44 원")
