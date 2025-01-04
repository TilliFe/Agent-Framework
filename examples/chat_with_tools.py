from framework import *
from .utils import *
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv 
import google.generativeai as genai
load_dotenv() 


####################################################################################################
# Define the Parts of the Prompt
####################################################################################################
# Define the system, i.e. the role of the AI
system = "You are a conversational agent that can answer questions, execute code, and check the weather."

# Define the tools available to the AI
tools = json.dumps(tool(tavily_search).tool_json)
tools += "\nand\n"
tools += json.dumps(tool(execute_python_code).tool_json)
tools += "\nand\n"
tools += json.dumps(tool(check_weather).tool_json)
guidelines = """
**Tool Use Guidelines:**
- Use tools only when necessary
- MUST include clear user-facing summaries in "content" field explaining:
    - What tools you're using and why
    - Interpretation of tool outputs
    - Final conclusions/recommendations
- Execute tools sequentially in separate messages
- Follow exact argument types from schema:
    - Numbers as integers/floats (not strings)
    - Arrays with correct element types  
    - Booleans as true/false
    - No expressions, only literal values
- Fix any type errors by adjusting arguments
"""
tool_usage = f"""
**You have access to the following tools:**
{tools}
{guidelines}
"""


# Define the output structure
class ToolCall(BaseModel):
    name: str = Field(..., description="The name of the tool.")
    arguments: Dict[str, str] = Field(
        {}, description="LITERAL_VALUE arguments for the tool."
    )


class Output(BaseModel):
    role: str = "AI"
    content: str = Field(
        ...,
        description="Your answer, summarizing response or reasoning in Markdown format",
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        None,
        description="The tool calls made by the conversational agent, can be null. The use can not see this field.",
    )


output_schema = Output.model_json_schema()
output_guidelines = f"""
Your responses MUST be written as a valid JSON object in the following structure: 
{output_schema}
Violating these requirements will cause errors. Fix by adjusting the response format.
"""


####################################################################################################
# Define the node functions
####################################################################################################
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
    )
    llm = LLM(model, lambda model, query: model.generate_content(query).text)
except Exception as e:
    raise e


def create_ai_response(state: Dict) -> Dict:
    """Creates an AI response using the current conversation state."""
    prompt = f"{system}\n{tool_usage}\n**Current Message History:**{state["messages"]}\n{output_guidelines}"
    ai_message = extract_json(llm.invoke(prompt))
    return {"messages": [ai_message]}


def execute_tools(state: Dict) -> Dict:
    """Executes the tools requested by the AI and returns their responses."""
    print()
    ai_message = state["messages"][-1]
    tool_calls = ai_message["tool_calls"]
    tool_responses = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        speach_bubble(
            f"Calling tool: {tool_name}",
            color="rgb(70,70,70)",
            padding=(0, 37, 0, 37),
            use_panel=False,
        )
        arguments = tool_call["arguments"]
        if tool_name == "tavily_search":
            result = tavily_search(**arguments)
        elif tool_name == "execute_python_code":
            result = execute_python_code(**arguments)
        elif tool_name == "check_weather":
            result = check_weather(**arguments)
        else:
            raise ValueError(f"Tool {tool_name} not found.")
        tool_responses.append({"name": tool_name, "result": result})
    return {"messages": [{"role": "tools", "content": tool_responses}]}


def create_user_response(state: Dict) -> Dict:
    """Gets input from the user and formats it as a message."""
    print()
    print("\033[94m>\033[0m ", end="")
    query = input()
    print("\033[A\033[K", end="")  # Clear the input line
    speach_bubble(query, bgcolor="rgb(50,50,50)")
    print()
    return {"messages": [{"role": "user", "content": query}]}


def check_ai_output(state: Dict) -> str:
    """Validates AI output and determines the next action based on the response."""
    ai_message = state["messages"][-1]
    validated, error = is_valid_json(ai_message, Output)
    if not validated:
        state["messages"].append(
            {"role": "output_checker", "content": f"Error: {error}"}
        )
        return "ai_node"
    elif ai_message["tool_calls"] is not None:
        return "tool_node"
    else:
        speach_bubble(ai_message["content"])
        return "user_node"


def determine_next_node(state: Dict) -> str:
    """Determines the next node based on user input commands (stop/reset)."""
    if state["messages"][-1]["content"] == "stop":
        return "END"
    if state["messages"][-1]["content"] == "reset":
        state["messages"] = []
        return "user_node"
    return "ai_node"


def END_response(state: Dict):
    """Handles the termination of the conversation."""
    return None


####################################################################################################
# Define the graph
####################################################################################################
def chat_with_tools():
    """Sets up and runs an interactive chat system with tool execution capabilities."""

    # Define the state schema
    class State(TypedDict):
        messages: List[Dict[str, str]]

    state_graph = Graph(
        state_schema=State, update_mechanisms={"messages": append_to_state}
    )

    # Define the nodes
    ai_node = Node("ai_node", create_ai_response)
    tool_node = Node("tool_node", execute_tools)
    user_node = Node("user_node", create_user_response)
    end_node = Node("END", END_response)

    # Add the nodes to the graph
    state_graph.add_node(user_node)
    state_graph.add_node(ai_node)
    state_graph.add_node(tool_node)
    state_graph.add_node(end_node)
    state_graph.set_start_node(user_node)
    state_graph.set_end_nodes(end_node)

    # Add the edges to the graph
    state_graph.add_conditional_edges(
        user_node,
        determine_next_node,
        {"ai_node": ai_node, "END": end_node, "user_node": user_node},
    )
    state_graph.add_conditional_edges(
        ai_node,
        check_ai_output,
        {"ai_node": ai_node, "user_node": user_node, "tool_node": tool_node},
    )
    state_graph.add_edge((tool_node, ai_node))

    # run the graph
    state_graph.invoke({"messages": []}, max_depth=None)
