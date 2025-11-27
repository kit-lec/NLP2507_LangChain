# 학습목표:
# Streamlit 의 data flow 와 data 가 처리되는 방식

# Streamlit 에선 'widget 에 입력한 data 가 변경'될때 마다 python 파일 '전체'가 다시 실행된다. (py 파일 위에서부터 아래까지 전부 다시 실행)
# 가령 사용자가 무언가를 입력하거나 slider 를 드래그 해서 data 가 변경될때마다 ..

import streamlit as st
import numpy as np
import pandas as pd

import os
import time

from dotenv import load_dotenv
load_dotenv()

print(f'✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...')

# 다양한 입력 widgets 들
#    https://docs.streamlit.io/develop/api-reference/widgets


st.title(time.strftime('%Y-%m-%d %H:%M:%S'))  # 확인해보자, 새로고침도.

model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
st.markdown(f'model: :green[{model}]')


name = st.text_input("What is your name?")
st.markdown(f'name: :green[{name}]')


value = st.slider('temperature', min_value=0.1, max_value=1.0)
st.write(value)

if model == 'GPT-3':
    st.write('Cheap')
else:
    st.write('expensive')
    country = st.text_input('what is your country?')
    st.write(country)

button = st.button('버튼을 눌러보세요') # => bool 리턴

if button:
    st.write(":blue[버튼]이 눌렸습니다 :sparkless:")

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
})

st.download_button(
    label='CSV로 다운로드',
    data=df.to_csv(), 
    file_name="sample.csv",
    mime='text/csv',
)

agree = st.checkbox("동의 하시겠습니까?") # => bool 리턴

if agree:
    st.write("동의 해주셔서 감사합니다 :100:")


mbti = st.radio(
    label="당신의 MBTI는 무엇입니까?",
    options=('ISTJ', 'ENFP', '선택지 없슴'),
)

st.write({
    'ISTJ': '당신은 :blue[현실주의자] 입니다',
    'ENFP': '당신은 :green[활동가] 입니다',
    '선택지 없슴': '당신에 대해 :red[알고 싶어요]:grey_exclamation:'
}[mbti])


mbti = st.selectbox(
    label='당신의 MBTI는 무엇입니까?',
    options=('ISTJ', 'ENFP', '선택지 없슴'),
    index = 2,
)

st.write({
    'ISTJ': '당신은 :blue[현실주의자] 입니다',
    'ENFP': '당신은 :green[활동가] 입니다',
    '선택지 없슴': '당신에 대해 :red[알고 싶어요]:grey_exclamation:'
}[mbti])


options = st.multiselect(  # => list 리턴
    label="당신이 좋아하는 과일은 뭔가요?",
    options=['망고', '오렌지', '사과', '바나나'],
    default=['망고', '오렌지',]
)

st.write(f'당신의 선택은: :red[{options}] 입니다.')


values = st.slider(
    label='범위의 값을 다음과 같이 지정할 수 있어요:sparkles:',
    min_value=0.0,
    max_value=100.0,
    value=(25.0, 75.0),  # 이 경우 slider() 는 tuple 리턴
)
st.write('선택 범위:', values)

# datetime 을 slider 구간에 사용 가능
from datetime import datetime as dt
import datetime

meeting_time  = st.slider(
    label="언제 약속을 잡는 것이 좋을까요?",
    min_value=dt(2020, 1, 1, 0, 0),
    max_value=dt(2020, 1, 7, 23, 0),
    value=dt(2020, 1, 3, 12, 0),
    step=datetime.timedelta(hours=1),
    format='MM/DD/YY - HH:mm',
)
st.write("선택한 약속 시간:", meeting_time)

title = st.text_input(
    label='가고 싶은 여행지가 있나요?',
    placeholder='여행지를 입력해 주세요',

)
st.write(f'당신이 선택한 여행지: :violet[{title}]')

age = st.number_input(
    label = '나이를 입력해 주세요',
    min_value=10,
    max_value=100,
    value=30,
    step=5,  
)
st.write('입력하신 나이는:', age)

import random
import datetime

st.markdown('---')
st.title(':sparkles: 로또 생성기:sparkles:')

def generate_lotto():
    lotto = [i + 1 for i in range(45)]
    random.shuffle(lotto)

    return lotto[:6]

button = st.button('로또를 생성해 주세요!')

if button:
    for i in range(1, 6):
        st.subheader(f'{i} 행운의 번호: :green[{generate_lotto()}]')

# 파일 업로더 위젯
st.markdown('---')
st.title('파일 업로드:sparkles:')

# ----------------------------------------------
# st.file_uploader() => None | UploadedFile | list of UploadedFile 리턴
# 파일 업로드 버튼 (업로드 기능)

file = st.file_uploader(
    '파일 선택(csv or excel)',
    type=['csv', 'xls', 'xlsx'],
)

if file:
    ext = file.name.split('.')[-1]  # 확장자
    if ext == 'csv':
        df = pd.read_csv(file)
        st.dataframe(df)