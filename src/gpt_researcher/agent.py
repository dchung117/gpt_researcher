from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, AgentExecutor

import config

load_dotenv()

LLM = ChatOpenAI(temperature=config.TEMPERATURE)

def create_agent(tools: list[str]):
    """
    Create GPT agent w/ access to provided tools

    Args
    ----
        tools: list[str]
            List of tools for agent to use in deployment.

    Returns
    -------
        AgentExecutor:
            Agent for Q&A using GPT and provided tooling.
    """
    tools = load_tools(tools)
    return initialize_agent(
        tools,
        LLM,
        max_iterations=config.MAX_ITERATIONS,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )