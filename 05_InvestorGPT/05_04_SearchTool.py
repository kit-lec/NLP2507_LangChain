from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.tools.simple import Tool
from langchain_core.tools.base import BaseTool

from pydantic import BaseModel, Field
from typing import Any, Type

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1)


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

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
    ],
)

prompt = "Give me information on Cloudflare's stock and help me analyze if it's a potential good investment.  Also tell me what symbol does the stock have"

agent.invoke(prompt)
