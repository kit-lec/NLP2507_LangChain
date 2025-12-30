from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.tools.simple import Tool
from langchain_core.tools.base import BaseTool  # 커스텀 Tool 클래스 정의를 위한 Base

from pydantic import BaseModel, Field
from typing import Any, Type  # typing 은 Python 의 기본 내장모듈

llm = ChatOpenAI(temperature=0.1, model='gpt-4o')


class CalculatorToolArgsSchema(BaseModel):
    a: float = Field(description="The first number")
    b: float = Field(description="The second number")


class CalculatorTool(BaseTool):
    name: Type[str] = "CalculatorTool"   # ✔tool 이름에 공백 있으면 안됨!
    description: Type[str] = """
    Use this to perform sums of two numbers.
    The first and second arguments should be numbers.
    Only receives two arguments.
    """

    # argument 의 스키마 정의
    # 이를 정의하기 위해 별도의 pydantic model 을 준비해보자.
    args_schema: Type[CalculatorToolArgsSchema] = CalculatorToolArgsSchema

    # 이 툴이 사용되렀을때 실행할 코드
    # def _run(self, *args: Any, **kwargs: Any) -> Any:
    def _run(self, a, b):
        return a + b

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=[CalculatorTool()],  # 커스텀 tool 지정
)

prompt = "Cost of $355.39 + $924.87 + $721.2 + $1940.29 + $573.63 + $65.72 + $35.00 + $552.00 + $76.16 + $29.12"

agent.invoke(prompt)







