from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny."

model = init_chat_model("groq:qwen/qwen3-32b")
model_with_tools = model.bind_tools([get_weather])

def basic_tool_call():
    response = model_with_tools.invoke("What's the weather like in New York?")
    print(f"################### Response: {response}")
    for tool_call in response.tool_calls:
        print(f"Tool called: {tool_call['name']} with arguments {tool_call['args']}")

def tool_execition_loops():
    messages = [{"role": "user", "content": "What's the weather like in New York?"}]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        tool_result = get_weather.invoke(tool_call)
        messages.append(tool_result)

    final_response = model_with_tools.invoke(messages)
    print(f"Final response: {final_response.text}")


if __name__ == "__main__":
    #basic_tool_call()
    tool_execition_loops()