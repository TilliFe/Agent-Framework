from mini import *
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
system = "You are a conversational assistant that shall reason and plan before giving an answer in a chain of thought. You have access to tools to help you. ALWAYS think step by step and explain your reasoning in a clear and understandable manner."

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
    thought: str = Field(
        ...,
        description="The chain of thought that will be used to generate the response. In Markdown format",
    )
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
    """Creates an AI response using the current conversation state."""
    prompt = f"{system}\n{tool_usage}\n**Current Message History:**{state["messages"]}\n{output_guidelines}\nYou must ONLY provide the next message. Do NOT provide the output schema again."
    llm_reply = llm.invoke(prompt)
    ai_message = extract_json(llm_reply)
    thought = ai_message["thought"]
    if state["show_cot"]:
        speach_bubble(
            f"Thought: {thought}",
            color="rgb(64,50,80)",
            padding=(0, 35, 0, 35),
            use_panel=True,
            use_markdown=True,
        )
    return {"messages": [ai_message]}


def execute_tools(state: Dict) -> Dict:
    """Executes the tools requested by the AI assistant."""
    ai_message = state["messages"][-1]
    tool_calls = ai_message["tool_calls"]
    tool_responses = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        speach_bubble(
            f"Calling tool: {tool_name}",
            color="rgb(64,50,80)",
            padding=(0, 35, 0, 35),
            use_panel=True,
            use_markdown=True,
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
    """Gets and formats user input from the console."""
    print()
    print("\033[94m>\033[0m ", end="")
    query = input()
    print("\033[A\033[K", end="")  # Clear the input line
    speach_bubble(query, bgcolor="rgb(50,50,50)")
    print()
    return {"messages": [{"role": "user", "content": query}]}


def check_ai_output(state: Dict) -> str:
    """Validates AI output and determines the next action based on tool calls."""
    ai_message = state["messages"][-1]
    validated, error = is_valid_json(ai_message, Output)
    if not validated:
        state["messages"].append(
            {"role": "output_checker", "content": f"Error: {error}"}
        )
        return "ai_node"
    elif "tool_calls" in ai_message and ai_message["tool_calls"] is not None:
        return "tool_node"
    else:
        speach_bubble(ai_message["content"])
        return "user_node"


def determine_next_node(state: Dict) -> str:
    """Determines the next node based on special commands or continues normal flow."""
    if state["messages"][-1]["content"] == "show CoT":
        state["show_cot"] = True
        return "user_node"
    elif state["messages"][-1]["content"] == "hide CoT":
        state["show_cot"] = False
        return "user_node"
    elif state["messages"][-1]["content"] == "stop":
        return "END"
    elif state["messages"][-1]["content"] == "reset":
        state["messages"] = []
        return "user_node"
    return "ai_node"


def END_response(state: Dict):
    """Handles the end of conversation state."""
    return None


####################################################################################################
# Define the graph
####################################################################################################
def define_CoT_graph():
    """Defines the conversation flow graph with all nodes and edges."""

    # Define the state schema
    class State(TypedDict):
        messages: List[Dict[str, str]]
        show_cot: bool = False

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

    return state_graph


def CoT_chat():
    """Initializes and runs the Chain of Thought chat system."""
    # Create a TraceLog instance
    trace_log = TraceLog()
    trace_callback_handler = TraceCallbackHandler(trace_log)
    callback_manager = CallbackManager([trace_callback_handler])

    # Define the chain with the graph and invoke it
    graph = define_CoT_graph().compile()
    input = {"messages": [], "show_cot": True}
    chain = graph | RunnableLambda(lambda x: x["messages"])
    final_state = chain.invoke(input, callbacks=callback_manager)

    # Access the trace information afterwards
    for log_entry in trace_log.get_logs():
        print()
        print(log_entry)
