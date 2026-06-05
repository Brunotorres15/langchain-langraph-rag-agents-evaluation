from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.types import Command


from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass

import os
from dotenv import load_dotenv
load_dotenv()

## Middleware allows you to intercept and modify the input and output of the LLM,
## enabling you to add custom logic, such as logging, error handling, or data transformation,
## before the LLM processes the input or after it generates the output.

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

### Summarization Middleware
# Automatically summarize the conversation history before sending it to the LLM,
# ensuring that the model receives a concise version of the conversation, which can help improve response quality.
### Messagebased Summarization Middleware
def messagebased_summarization_middleware():
    agent = create_agent("gpt-4o-mini",
                    checkpointer=InMemorySaver(),
                    middleware=[
                        SummarizationMiddleware(
                            model="gpt-4o-mini",
                            trigger=("messages", 10), #Trigger the summarization after every 10 messages in the conversation history
                            keep=("messages",4) #Keep the last 4 messages in the conversation history when summarizing
                            )
                        ]
                    )
    

    ### Run with a thread id
    config={"configurable": {"thread_id": "test-1"}}

    questions = [
        "What is 2+2?",
        "What is 10+5?",
        "What is 15+7?",
        "What is 20+10?",
        "What is 25+15?",
        "What is 30+20?"
    ]

    for q in questions:
        response = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
        print(f"Messages: {response}")
        print(f"Messages: {len(response['messages'])}")


###
## Token Size
####
@tool
def search_hotel(city: str) -> str:
    """Search for hotels in a given city and return a list of hotel names."""
    return f"""Found hotels in {city}: 
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 3 star, $150/night, free breakfast, wifi
    3. Budget Stay - 2 star, $80/night, free parking, wifi
    """

def count_tokens(messages):
    total_chars = sum(len(str(message.content)) for message in messages)
    return total_chars // 4 # 4 chars = 1 token (approximation)

def token_size_middleware():
    agent = create_agent("gpt-4o-mini",
                    checkpointer=InMemorySaver(),
                    tools=[search_hotel],
                    middleware=[
                        SummarizationMiddleware(
                            model="gpt-4o-mini",
                            trigger=("tokens", 550), #Trigger the summarization after every 550 tokens in the conversation history
                            keep=("tokens",200) #Keep the last 200 tokens in the conversation history when summarizing
                            )
                        ]
                    )
    
    config={"configurable": {"thread_id": "test-2"}}
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]

    for city in cities:
        response = agent.invoke({"messages": [HumanMessage(content=f"Search for hotels in {city}")]},
                                config= config
                                )
        
        tokens = count_tokens(response["messages"])
        print(f"{city}: ~{tokens} tokens, {len(response['messages'])} messages\n")
        print(f"Response: {response['messages']}\n")

###
## Human in the Loop Middleware
# Allows you to involve a human in the decision-making process of the agent,
# enabling you to review and approve the agent's actions before they are executed, 
# which can be particularly useful for sensitive tasks or when you want to ensure the quality of the agent's decisions.
###

def read_email_tool(email_id: str) -> str:
    """Read the content of an email given its ID."""
    return f"Email content for Email ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient with a given subject."""
    return f"Email sent to {recipient} with subject '{subject}'"

def human_in_the_loop_middleware():
    agent = create_agent("gpt-4o-mini",
                    checkpointer=InMemorySaver(),
                    tools=[read_email_tool, send_email_tool],
                    middleware=[
                        HumanInTheLoopMiddleware(
                            interrupt_on={
                                "send_email_tool":{
                                    "allowed_decisions":["approve", "reject", "edit"] #Allowed decisions for the human reviewer when the agent decides to use the send_email_tool"
                                },
                                "read_email_tool": False,
                            }, #Trigger the human review after every 5 messages in the conversation history
                            )
                        ]
                    )
    
    config={"configurable": {"thread_id": "test-3"}}
    response = agent.invoke({"messages": [HumanMessage(content="Send email to john@example.com with subject 'Meeting Reminder' and body 'Hey, how are you?'")]},
                            config=config
                            )
    #print(f"Response: {response}\n")

    if "__interrupt__" in response:
        print(f"Paused | Approving...")

        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            ),
            config=config
        )
        print(f"Result: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    #messagebased_summarization_middleware()
    #token_size_middleware()
    human_in_the_loop_middleware()