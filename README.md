# LangChain-Mini

Welcome to **Mini** - a minimalistic toolkit for building sophisticated, Graph-based AI agents.

This project serves an educational purpose: to demystify the current hype around AI agents by showing how to prompt LLMs to use tools, generate structured outputs, and connect to real-world applications from scratch. Most architectural decisions for this project were inspired by LangChain and LangGraph.

You can find the source code is in the [framework](https://github.com/TilliFe/LangChain-Mini/tree/main/framework) directory. For a complete overview of the Framework's components and a simple CoT agent (with tools), check out [framework_explained.ipynb](https://github.com/TilliFe/LangChain-Mini/blob/main/framework_explained.ipynb).

### Key Features
* **Tool Integration 🛠️**: Turn any Python function into a LLM compatible tool
* **Structured Outputs**: Get structured outputs from LLMs for easy parsing
* **Prompt Templates**: Systematic prompt generation for consistent agent behavior
* **Agents as State Machines**: Create complex graph-based agents with ease
* **Tracing and Logging**: A unified module interface for easy debugging

## Quick Start 🚀

1. **Installation**
    ```bash
    git clone https://github.com/TilliFe/LangChain-Mini.git
    cd LangChain-Mini
    pip install -r requirements.txt
    ```

2. **Configuration**

    Create `.env` file:
    ```bash
    GEMINI_API_KEY="YOUR_KEY"   # from https://aistudio.google.com/apikey
    TAVILY_API_KEY="YOUR_KEY"   # from https://app.tavily.com/home
    ```

3. **Run Examples**
    ```bash
    python run_examples.py
    ```

    You can now interact with the tool-using AI Agent directly in your console, as shown in the screenshot below. The default LLM used for this Agent is Gemini 2.0 Flash. You can easily switch the model and customize the Agent's behavior in the [/examples/CoT_chat.py](https://github.com/TilliFe/LangChain-Mini/blob/main/examples/CoT_chat.py) file.

    ![alt text](/assets/CoT_example_screenshot.png)

## Happy Building! 🔥
