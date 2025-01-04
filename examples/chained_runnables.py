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

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
    )
    llm = LLM(model, lambda model, query: model.generate_content(query).text)
except Exception as e:
    raise e


class ToolCall(BaseModel):
    """Represents a tool invocation with a name and arguments."""

    name: str = Field(..., description="The name of the tool.")
    arguments: Dict[str, str] = Field(
        {}, description="LITERAL_VALUE arguments for the tool."
    )


class Output(BaseModel):
    """Defines the structure for AI assistant responses."""

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


def chained_runnables():
    """Demonstrates the use of chained runnables with branching logic based on keywords."""
    output_schema = Output.model_json_schema()
    output_guidelines = f"""
    Your responses MUST be written as a valid JSON object in the following structure: 
    {output_schema}
    Violating these requirements will cause errors. Fix by adjusting the response format.
    """

    prompt = RunnableChatPrompt(
        system="You are a helpful AI assistant.",
        tools="You do not have access to any tools.",
        message_history=[],
        output=output_guidelines,
    )

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
        )
        llm = LLM(model, lambda model, query: model.generate_content(query).text)
    except Exception as e:
        raise e

    chain = (
        prompt
        | llm
        | RunnableLambda(lambda ai_message: ai_message)
        | RunnableBranch(
            (lambda x: "joke" in x, lambda x: "A joke was told."),
            (lambda x: "why" in x, lambda x: "The ai is confused."),
        )
    )

    output = chain.invoke("Say a sentence wehre teh word joke comes up twice.")
    print(output)

    output = chain.invoke("Say a sentence wehre teh word 'why' comes up twice.")
    print(output)
