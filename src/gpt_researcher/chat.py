import chainlit as cl
from langchain.agents import load_tools

from .agent import LLM, create_agent

@cl.on_chat_start
def start():
    agent = create_agent(["arxiv"])

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(msg: str):
    agent = cl.user_session.get("agent")
    callback = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(msg, callbacks=[callback])