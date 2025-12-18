from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
import requests

#init the model
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.7,
)


@tool
def get_weather_for_location(city: str) -> str:
    """Get the weather for a specific city."""
    try:
        url = f"http://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data['current_condition'][0]
        temp = current['temp_C']
        desc = current['weatherDesc'][0]['value']

        return f"The weather in {city} is {desc} with a temperature of {temp}^C"
    except:
        return f"Unable to fetch weather for {city}"


@dataclass
class Context:
    """Custom runtime contexr schema."""
    user_id:str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user IP address"""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        data = response.json()
        return data.get('city', 'Unknown')
    except:
        return "Unable to determin location."




@dataclass
class ResponseFormat:
    """Response schema for the agent"""
    punny_response: str
    weather_conditions: str | None = None

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks seriously.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema= Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# 'thread_id' is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content" : "What is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['messages'][-1].content)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['messages'][-1].content)