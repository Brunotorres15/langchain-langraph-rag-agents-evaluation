from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

import os
from dotenv import load_dotenv
load_dotenv()



os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = init_chat_model("groq:qwen/qwen3-32b")

def text_prompt():

    # Text Prompt.
    response = model.invoke("what is langchain?")
    print(response)

def messages_prompt():
    # Messages Prompt.
    messages = [
        SystemMessage(content="You are a poetry expert."),
        HumanMessage(content="Write a poem on Brazilian culture."),
    ]
    response = model.invoke(messages)
    print(response.content)

# Detailed instructions to the LLM through System Messages.
def detailed_system_message_prompt():
    messages = [
        SystemMessage(content="You are a helpful Senior Python developer. " \
        "You will be given a task and you should provide a detailed step-by-step solution to the problem with code example."),
        HumanMessage(content="How do I create a REST API using FastAPI?"),
    ]
    response = model.invoke(messages)
    print(response.content)

# AI Message
def ai_message_prompt():
    ai_msg = AIMessage("I am an AI language model created by Bruno")

    messages = [
        SystemMessage("You are a helpful Assistant that always answers in a friendly manner."),
        HumanMessage("Can you help me?"),
        ai_msg,
        HumanMessage("Great, so whats 3 + 2 ?")
   ]

    response = model.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    ai_message_prompt()
    
