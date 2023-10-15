from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, AgentExecutor, Tool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import BaseTool, PythonREPLTool, YouTubeSearchTool
from .settings import config

load_dotenv()

LLM = ChatOpenAI(temperature=config.TEMPERATURE,
    streaming=True)

def get_tools(tools: list[str]) -> list[BaseTool]:
    """
    Parse and return list of tools.

    Args
    ----
        tools: list[str]
            List of tools for agent to use in deployment.

    Returns
    -------
        list[BaseTool]:
            List of tools for chat agent to use.
    """
    other_tools = []
    if "youtube_search" in tools:
        other_tools.append(
            Tool(
            name="Youtube Search",
            func=YouTubeSearchTool().run,
            description="Useful for when you need to provide links to Youtube videos. Put prefix https://www.youtube.com/ in front of the link."
        )
        )
        yt_idx = tools.index("youtube_search")
        tools.pop(yt_idx)

    common_tools = []
    if tools:
        common_tools = load_tools(tools, llm=LLM)
    return common_tools + other_tools

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

    tools = get_tools(tools)
    return initialize_agent(
        tools,
        LLM,
        max_iterations=config.MAX_ITERATIONS,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )