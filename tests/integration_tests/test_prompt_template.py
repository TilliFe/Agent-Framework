from mini import *
import json
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv 
import google.generativeai as genai
load_dotenv()

def add_numbers(a: int, b: int) -> int:  # Changed from do_sth_random
    """Adds two numbers."""
    return a + b


def subtract_numbers(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b


def test_prompt_template():
    """Tests the prompt template functionality."""
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
        )
        llm = LLM(model, lambda model, query: model.generate_content(query).text)
    except Exception as e:
        raise e

    # Define the prompt for zero-shot learning
    examples = [
        HumanMessage("What is 7 - 3?")(),
        AIMessage(
            "Um 7 - 3 zu berechnen, verwende ich das Werkzeug subtract_numbers.",
            [{"name": "add_numbers", "arguments": {"a": 2, "b": 2}}],
        )(),
        ToolMessage("4")(),
        AIMessage("Die Antwort ist 4.")(),
        HumanMessage("What is 5 + 7?")(),
        AIMessage(
            "Um 5 + 7 zu berechnen, verwende ich das Werkzeug add_numbers.",
            [{"name": "add_numbers", "arguments": {"a": 5, "b": 7}}],
        )(),
        ToolMessage("12")(),
        AIMessage("Die Antwort ist 12.")(),
    ]

    # Define the prompt
    prompt = PromptTemplate.from_messages(
        messages=[
            ("system", "You are a helpful AI assistant. Always answer in German."),
            ("example", examples),
            ("tools", [tool(add_numbers), tool(subtract_numbers)]),
            ("structured_output", "int"),
        ]
    )

    # human query
    query = HumanMessage("What is 6 plus 7?")()
    print(query)

    # call model
    response = llm.invoke(prompt + HumanMessage("What is 6 plus 7?")())
    print(response)

    # call tool
    if response["tool_calls"]:
        tool_name = response["tool_calls"][0]["name"]
        tool_arguments = response["tool_calls"][0]["arguments"]
        if tool_name == "add_numbers":
            result = add_numbers(**tool_arguments)
        tool_message = ToolMessage(result)()
        print(tool_message)
    else:
        tool_message = ToolMessage("Error calling tool:" + tool_name)()

    # call model again with tool response
    response = llm.invoke(
        prompt + HumanMessage("What 6 plus 7?")() + json.dumps(response) + tool_message
    )
    print(response)
