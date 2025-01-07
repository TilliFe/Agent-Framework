from mini import (
    Chat,
    LLMModel,
    tool,
    execute_python_code,
    check_weather,
    tavily_search,
)
from pydantic import BaseModel


class WeatherOutput(BaseModel):
    location: str
    temperature: float
    unit: str = "Celsius"


class AllOutputs(BaseModel):
    outputs: list[WeatherOutput]


def test_weather_chat():
    weather_chat = Chat(
        llm_model=LLMModel(),
        tools=[tool(execute_python_code), tool(check_weather), tool(tavily_search)],
        structured_output=AllOutputs,
    )
    response = weather_chat.invoke(
        "What is the weather in both Stuttgart and Berlin right now? Use your tools."
    )
