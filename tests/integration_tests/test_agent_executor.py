from framework import *
import google.generativeai as genai
import os
from dotenv import load_dotenv 
load_dotenv()

def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


def subtract_numbers(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b


def test_react_agent():
    """Tests the prompt template functionality."""
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
        )
        llm = LLM(model, lambda model, query: model.generate_content(query).text)
    except Exception as e:
        raise e

    llm = AgentExecutor(
        llm_model=llm,
        tools=[tool(add_numbers), tool(subtract_numbers)],
        messages=[],
        structured_output=None,
    )

    query = "Can you add 20 and 30 and then subtract 10?"
    response = llm.invoke(query, print_intermediate_steps=True)
    print(response)
