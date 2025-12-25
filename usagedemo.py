############### static model#############
from langchain.agents import create_agent

agent = create_agent("gpt-5",tools=tools)


################# use the model package###########
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_completion_tokens=10000,
    timeout=30,
    #....
)

agent = create_agent(model=model, tools=tool)


############ dynamic models with the control of different stages###############
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt_model")
advanced_model = ChatOpenAI(model="got-40")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model= basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)

####################### Tools ######################
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """search for information"""
    return f"Results for:{query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"Weather in {location}: Sunny, 72'F"

agent = create_agent(model, tools=[search, get_weather])

################## Tool error handling ##################
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """handle tool execution errors with custom messages"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: please check your input and try again.({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
    
agent = create_agent(
    model="gpt-40",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)

############ System prompt (with str) #################
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)


########### System prompt (with SystemMessage) ############
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text":"You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type":"text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control":{"type": "ephemeral"}
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
)

###########  Dynamic system prompt ############
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role:str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.contextget("user_role", "user")
    base_prompt = "You are a helpful assistant,"

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."
    
    return base_prompt

agent = create_agent(
    model="gpt-40",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content":"Explain machine learning"}]},
    context={"user_role": "expert"}
)



########  ToolStrategy  with structured output ############
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role":"user", "content":"Extract contact info from: John Doe, john@example.com, 123123123"}]
})

result["structured_response"]


################# ProvideStratrgy #####################
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)

######  state via middleware #####################3

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_perference: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state:CustomState, runtime) -> dict[str, Any] | None:
        121

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

result = agent.invoke({
    "messages":[{"role": "user", "content":"I prefer technical explanations"}],
    "user_perferences": {"style": "technical", "verbosity": "detailed"},
})


######## state_schema##################
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})