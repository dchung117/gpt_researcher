from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, AgentExecutor
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from .settings import config

load_dotenv()

LLM = ChatOpenAI(temperature=config.TEMPERATURE,
    streaming=True)

def create_agent(tools: list[str],
    write_code: bool = False):
    """
    Create GPT agent w/ access to provided tools

    Args
    ----
        tools: list[str]
            List of tools for agent to use in deployment.
        write_code: bool = False
            Flag for agent to run code.
    Returns
    -------
        AgentExecutor:
            Agent for Q&A using GPT and provided tooling.
    """
    if write_code:
        return create_python_agent(
            llm=LLM,
            tool=PythonREPLTool(),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

    tools = load_tools(tools, llm=LLM)
    return initialize_agent(
        tools,
        LLM,
        max_iterations=config.MAX_ITERATIONS,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )