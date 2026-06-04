from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")

## Pydantic is a data validation library that allows you to define data models with type annotations.
## It provides a way to validate and parse data, 
# ensuring that it conforms to the specified types and constraints. 
# Pydantic is often used in Python applications to handle data validation and serialization, 
# making it easier to work with structured data.
class Movie(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
    rating: float = Field(..., description="The rating of the movie out of 10")

# Structured output allows you to define a specific format for the response from the LLM,
# ensuring that the output adheres to a predefined structure, such as a Pydantic model
def structured_output_prompt():
    messages = [
        SystemMessage(content="You are a helpful assistant that provides information about movies. "),
        HumanMessage(content="Can you provide information about the movie Inception?"),
    ]

    model_with_structured_output = model.with_structured_output(Movie)
    response = model_with_structured_output.invoke(messages)
    print(response)

# Message output with parsed structure allows you to receive the response from the LLM in a structured format, 
# such as a Pydantic model, while also including the raw text response from the LLM.
def message_output_with_parsed_structure():
    messages = [
        SystemMessage(content="You are a helpful assistant that provides information about movies. "),
        HumanMessage(content="Can you provide information about the movie Inception?"),
    ]

    model_with_structured_output = model.with_structured_output(Movie, include_raw=True)
    response = model_with_structured_output.invoke(messages)
    print(response)

# Nested structures allow you to define complex data models that can represent hierarchical relationships between different entities.
def nested_structure():
    class Actor(BaseModel):
        name: str = Field(..., description="The name of the actor")
        age: int = Field(..., description="The age of the actor")

    class MovieDetails(BaseModel):
        title: str = Field(..., description="The title of the movie")
        director: str = Field(..., description="The director of the movie")
        release_year: int = Field(..., description="The year the movie was released")
        rating: float = Field(..., description="The rating of the movie out of 10")
        actors: list[Actor] = Field(..., description="A list of main actors in the movie")
        budget: float | None = Field(None, description="The budget of the movie in USD")

    messages = [
        SystemMessage(content="You are a helpful assistant that provides information about movies. "),
        HumanMessage(content="Can you provide information about the movie Inception including budget and its main actors?"),
    ]

    model_with_structured_output = model.with_structured_output(MovieDetails)
    response = model_with_structured_output.invoke(messages)
    print(response)


###
#       TypedDict is a type hint that allows you to specify the expected types of keys and values in a dictionary.
#       It is part of the typing module in Python and is used to provide type information for dictionaries, 
#       making it easier to understand the structure of the data and catch potential type errors during development.
###

def typed_dict_structure():
    class MovieDict(TypedDict):
        title: Annotated[str, ..., "The title of the movie"]
        director: Annotated[str, ..., "The director of the movie"]
        release_year: Annotated[int, ..., "The year the movie was released"]
        rating: Annotated[float, ..., "The rating of the movie out of 10"]

    model_with_typedict = model.with_structured_output(MovieDict)
    response = model_with_typedict.invoke("Please provide the details of the movie Avengers.")
    print(response)


###
#       DataClasses are a feature in Python that provides a decorator and functions for automatically adding special methods to user-defined classes.
#       They are used to create classes that primarily store data, 
#       and they automatically generate methods like __init__, __repr__, and __eq__ based on the class attributes, 
#       making it easier to create and manage data objects.
###
def agent_data_class():

    @dataclass
    class ContactInfo:
        """A dataclass to represent contact information of a person."""
        name: str 
        email: str 
        phone: str 

    agent = create_agent("gpt-5",
                         response_format=ContactInfo
                         )

    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Extract contact information from the following text: 'John Doe, email: john.doe@example.com, phone: 123-456-7890'"
        }]
    })
    print(response["structured_response"])

if __name__ == "__main__":
    #structured_output_prompt()
    #message_output_with_parsed_structure()
    #nested_structure()
    #typed_dict_structure()
    agent_data_class()