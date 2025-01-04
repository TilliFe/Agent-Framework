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
system = "You are a conversational agent. You do not have access to any tools."


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
        description="The tool calls made by the conversational agent, can be null.",
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
    """Generates an AI response based on the current conversation state."""
    prompt = f"{system}\n**Current Message History:**{state["messages"]}\n{output_guidelines}"
    ai_message = extract_json(llm.invoke(prompt))
    return {"messages": [ai_message]}


def create_user_response(state: Dict) -> Dict:
    """Captures and formats user input from the console."""
    print()
    print("\033[94m>\033[0m ", end="")
    query = input()
    print("\033[A\033[K", end="")  # Clear the input line
    speach_bubble(query, bgcolor="rgb(50,50,50)")
    print()
    return {"messages": [{"role": "user", "content": query}]}


def check_ai_output(state: Dict) -> str:
    """Validates the AI output against the expected schema and format."""
    ai_message = state["messages"][-1]
    validated, error = is_valid_json(ai_message, Output)
    if not validated:
        state["messages"].append(
            {"role": "output_checker", "content": f"Error: {error}"}
        )
        return "ai_node"
    else:
        speach_bubble(ai_message["content"])
        return "user_node"


def determine_next_node(state: Dict) -> str:
    """Determines the next conversation flow based on user input."""
    if state["messages"][-1]["content"] == "stop":
        return "END"
    if state["messages"][-1]["content"] == "reset":
        state["messages"] = []
        return "user_node"
    return "ai_node"


def END_response(state: Dict):
    """Handles the conversation termination."""
    return None


####################################################################################################
# Define the graph
####################################################################################################
def basic_chat():
    """Sets up and runs a basic conversational agent with a state graph."""

    # Define the state schema
    class State(TypedDict):
        messages: List[Dict[str, str]]

    state_graph = Graph(
        state_schema=State, update_mechanisms={"messages": append_to_state}
    )

    # Define the nodes
    ai_node = Node("ai_node", create_ai_response)
    user_node = Node("user_node", create_user_response)
    end_node = Node("END", END_response)

    # Add the nodes to the graph
    state_graph.add_node(user_node)
    state_graph.add_node(ai_node)
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
        ai_node, check_ai_output, {"ai_node": ai_node, "user_node": user_node}
    )

    # run the graph
    state_graph.invoke({"messages": []}, max_depth=None)
