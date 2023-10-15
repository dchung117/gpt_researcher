import os
import chainlit as cl
from langchain.agents import load_tools

from .agent import LLM, create_agent

@cl.on_chat_start
def start():
    if os.environ.get("TOOLS"):
        tools = os.environ["TOOLS"].split(",")
    else:
        raise ValueError("You must set TOOLS as an environment variable.")
    agent = create_agent(tools)

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(msg: str):
    agent = cl.user_session.get("agent")
    callback = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(msg, callbacks=[callback])