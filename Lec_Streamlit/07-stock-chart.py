import streamlit as st
import FinanceDataReader as fdr
import datetime

print('✅ 실행됨')

# Finance Data Reader
# https://github.com/financedata-org/FinanceDataReader

start_date = st.date_input(
    '조회 시작일을 선택해 주세요',
    value = datetime.datetime(2025, 11, 1)
)

end_date = st.date_input(
    "조회 종료일을 선택해 주세요",
    value=datetime.datetime(2025, 11, 30)
)


code = st.text_input(
    '종목코드', 
    value='005930',  # 삼성전자
    placeholder='종목코드를 입력해 주세요'
)

print(f'\tcode:{code}, start_date:{start_date}, end_date:{end_date}') #  확인

if code and start_date and end_date:
    df = fdr.DataReader(code, start_date, end_date)
    st.dataframe(df)

    data = df.sort_index(ascending=True).loc[:, 'Close']
    st.line_chart(data)