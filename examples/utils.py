from rich.markdown import Markdown, Panel
from rich.padding import Padding
from rich.style import Style
from rich.console import Console
import json
import os
import requests
import subprocess


def speach_bubble(
    message,
    color="rgb(230,230,230)",
    bgcolor="rgb(30,30,30)",
    padding=(0, 35, 0, 35),
    use_panel=True,
    use_markdown=True,
):
    """Displays a formatted message in a speech bubble style using rich formatting."""
    console = Console()
    base_message = (
        message
        if not use_markdown
        else Markdown(message, style=Style(color=color, bgcolor=bgcolor))
    )
    base_message = (
        Panel(
            base_message,
            padding=(0, 1, 0, 1),
            border_style=Style(color=bgcolor),
            style=Style(color=color, bgcolor=bgcolor),
        )
        if use_panel
        else base_message
    )
    base_message = Padding(base_message, pad=padding, expand=True)
    console.print(base_message)


def add_numbers(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b


def subtract_numbers(a: int, b: int) -> int:
    """Subtracts one integer number from another."""
    return a - b


def execute_python_code(python_code: str) -> str:
    """Executes Python code and returns the output."""
    try:
        result = subprocess.run(
            ["python3", "-c", python_code], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr.strip()}"


def check_weather(city: str) -> str:
    """Gets the current weather for a specified city."""
    url = f"https://wttr.in/{city}?format=%C+%t"  # %C for weather condition, %t for temperature
    response = requests.get(url)

    if response.status_code == 200:
        return f"The current weather in {city} is: {response.text}"
    else:
        return f"Error fetching weather data for {city}. Please check the city name or try again later."


def plot_a_curve(x_values: list[float], y_values: list[float]) -> str:
    """Creates a simple line plot from x and y values."""
    plt.plot(x_values, y_values)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Curve Plot")
    plt.show()
    return "The Plot has been generated in the console."


def tavily_search(query: str, max_results: str = "5") -> str:
    """Performs a web search using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        return "Error: Tavily API key not found. Please set the TAVILY_API_KEY environment variable."

    try:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=tavily_api_key)
        search_result = tavily.search(query=query, max_results=max_results)

        formatted_results = [
            f"Title: {result.get('title', 'No Title')}\nURL: {result.get('url', 'No URL')}\nContent: {result.get('content', 'No Content')}\n"
            for result in search_result.get("results", [])
        ]

        return (
            "\n\n".join(formatted_results) if formatted_results else "No results found."
        )

    except ImportError:
        return "Error: tavily-python package is not installed. Please install it using 'pip install tavily-python'."
    except Exception as e:
        return f"An error occurred during Tavily search: {str(e)}"
