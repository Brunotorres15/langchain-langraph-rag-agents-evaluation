from urllib import response

from langchain.chat_models import init_chat_model

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def stream_response(model, prompt):
    ##### Grop Model
    groq_model = init_chat_model("groq:qwen/qwen3-32b")
    for chunk in groq_model.stream("Hello, are you alive?"):
        print(chunk.text, end="|", flush=True)


def batch_response():
    ##### Grop Model
    questions = [
        "Hello, are you alive?", 
        "What is the capital of France?", 
        "Who won the World Cup in 2018?"
        ]
    
    groq_model = init_chat_model("groq:qwen/qwen3-32b")
    responses = groq_model.batch(questions)
    
    for response in responses:
        print(response)

if __name__ == "__main__":
    batch_response()