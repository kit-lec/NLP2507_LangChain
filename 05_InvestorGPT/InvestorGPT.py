import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
print(f'\tALPHA_VANTAGE_API_KEY={alpha_vantage_api_key[:5]}...')

#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

import requests
from langchain_core.messages.system import SystemMessage
from langchain_openai.chat_models.base import ChatOpenAI

from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1, 
    model='gpt-4o',
)

# ────────────────────────────────────────
# ♒ Tools & Agent
# ────────────────────────────────────────

# 회사 심볼 tool
class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name: Type[str] = "StockMarketSymbolSearchTool"
    description: Type[str] = """
        Use this tool to find the stock market symbol for a company.
        It takes a query as an argument.
        Example query: Stock Market Symbol for Apple Company
        """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        # 검색기능 수행
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)  # 검색결과(들) 을 하나의 str 으로 묶어서 리턴.
    

# 회사 개요 tool

class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(
        description="Stock symbol of the company.Example: AAPL,TSLA",
    )


class CompanyOverviewTool(BaseTool):
    name: Type[str] = "CompanyOverview"
    description: Type[str] = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        time.sleep(1)  # free-plan 에선 1초당 1request 만 허용 (2025.12 현재)
        return r.json()  # 응답받은 json 을 파이썬 객체로 리턴.


# 손익계산서(income statement) 툴.
class CompanyIncomeStatementTool(BaseTool):
    name: Type[str] = "CompanyIncomeStatement"
    description: Type[str] = """
    Use this to get the income statement of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        time.sleep(1)  # free-plan 에선 1초당 1request 만 허용 (2025.12 현재)
        return r.json()["annualReports"]  # 응답받은 json 을 파이썬 객체로 리턴.


# 주가정보 툴
class CompanyStockPerformanceTool(BaseTool):
    name: Type[str] = "CompanyStockPerformance"
    description: Type[str] = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        time.sleep(1)  # free-plan 에선 1초당 1request 만 허용 (2025.12 현재)
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),  # 회사의 개요 정보
        CompanyIncomeStatementTool(), # 회사의 손익계산서 정보
        CompanyStockPerformanceTool(),  # 회사의 주간 주가 정보
    ],

    agent_kwargs={
        "system_message": SystemMessage(
            content = """
            You are a hedge fund manager.           
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.    
            Consider the performance of a stock, the company overview and the company income statement.           
            Be assertive in your judgement and recommend the stock or advise the user against it.
            """
        ),
    },
)



# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────



# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT
            
    Welcome to InvestorGPT.
            
    Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$")) # 확인용