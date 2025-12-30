from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
# ZERO_SHOT_REACT_DESCRIPTION 는 StructuredTool 을 사용하지 않고 그냥 Tool 을 사용한다.
# from langchain_core.tools.structured import StructuredTool
from langchain_core.tools.simple import Tool  # <- Tool


llm = ChatOpenAI(temperature=0.1, model='gpt-4o')

# Zero shot ReAct 는 한개의 매개변수만 가능!
# plus(10, 20) 형태가 아니라 plus("10,20") 형태로 한개의 매개변수에 담아 호출되도록 하자.
def plus(inputs):  
    a, b = inputs.split(",")
    return float(a) + float(b)

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[
        Tool.from_function(
            func=plus,
            name="Sum Calculator",
            # 
            description="Use this to perform sums of two numbers. Use this tool by sending a pair of numbers separated by a comma.\nExample:1,2",
        ),
    ],
)

prompt = "Cost of $355.39 + $924.87 + $721.2 + $1940.29 + $573.63 + $65.72 + $35.00 + $552.00 + $76.16 + $29.12"

agent.invoke(prompt)