from langchain.chat_models import init_chat_model

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


model = init_chat_model("gpt-4o-mini")

# invoke the model
#response = model.invoke("Hello, are you alive?")
#print(response)

##### Grop Model
groq_model = init_chat_model("groq:qwen/qwen3-32b")
response = groq_model.invoke("Hello, are you alive?")
print(response)