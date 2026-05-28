import os 
from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # This is a mock function. In a real implementation, you would call a weather API.
    return f"The current weather in {city} is sunny with a temperature of 25°C."

agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can answer questions about the world and perform tasks using tools."
)

try:
    response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in New York?"}]})
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print(response['messages'])