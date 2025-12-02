import streamlit as st
import time

st.title('st.progress')

with st.expander('이 앱에 대하여'):
    st.write('`st.progress` 명령어를 사용하여 Streamlit 앱에서 계산의 진행 상태를 표시할 수 있습니다.')

my_bar = st.progress(0)    

for i in range(100):
    time.sleep(0.05)
    my_bar.progress(i + 1)

st.balloons()    