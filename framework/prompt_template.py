import json
from typing import Callable, Dict, Optional, Any
from .runnable import Runnable, CallbackManager


class PromptTemplate(Runnable):
    """Template engine for generating structured prompts with system messages, tools, examples, and output formats."""

    def __init__(
        self,
        system=None,
        tools=None,
        tool_choices=None,
        structured_output=None,
        examples=None,
    ):
        self.system = (
            system if system is not None else "You are a helpful AI assistant."
        )
        self.tools = tools if tools is not None else []
        self.tool_choices = tool_choices if tool_choices is not None else []
        self.structured_output = structured_output
        self.examples = examples

    @staticmethod
    def from_messages(messages):
        """Creates a formatted prompt from a sequence of message tuples."""
        prompt_template = PromptTemplate()
        for key, value in messages:
            if key == "system":
                prompt_template.system = value
            elif key == "examples":
                prompt_template.examples = value
            elif key == "tools":
                prompt_template.tools = value
            # elif key == "structured_output":
            #     prompt_template.structured_output = value

        prompt = ""
        for key, value in messages:
            if key == "system":
                prompt += prompt_template._format_role_template()
            elif key == "examples":
                prompt += prompt_template._format_examples_template()
            elif key == "tools":
                prompt += prompt_template._format_tools_template()
            # elif key == "structured_output":
            #     prompt_template.structured_output = value
            else:
                if value is not None:
                    prompt += f"{key}: {value}\n"
                else:
                    prompt += f"{key}\n"

        prompt += prompt_template._format_output_template()

        return prompt

    def update(self, chat_history):
        """Updates and returns the complete prompt template with all sections combined."""
        # Build template sections
        role_template = self._format_role_template()
        tools_template = self._format_tools_template()
        examples_template = self._format_examples_template()
        history_template = self._format_history_template(chat_history)
        output_template = self._format_output_template()

        # Combine and filter empty sections
        return "\n".join(
            filter(
                None,
                [
                    role_template,
                    tools_template,
                    examples_template,
                    history_template,
                    output_template,
                ],
            )
        )

    def _format_examples_template(self):
        """Formats the examples section of the prompt."""
        if self.examples is None or len(self.examples) == 0:
            return ""

        examples_list = json.dumps(self.examples, indent=4)

        return f"""
        **Examples:**
        {examples_list}
        Do not copy-paste examples directly. Use them as a reference to structure your response.
        """

    def _format_role_template(self):
        """Formats the role definition section of the prompt."""
        return f"""
        Your role definition, NEVER switch your role:
        {self.system}
        """

    def _format_tools_template(self):
        """Formats the tools section with their schemas and usage guidelines."""
        if self.tools is None or len(self.tools) == 0:
            return "You CANNOT use any tools in this task."

        # List all tools with their schemas
        tools_list = self.format_tools_as_json()

        # Updated guidelines with explicit type instructions
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

        tool_choices = (
            ("You MUST use the following tools: " + ", ".join(self.tool_choices))
            if self.tool_choices
            else ""
        )

        return f"""
        **You have access to the following tools:**
        {tools_list}
        {guidelines}
        {tool_choices}
        """

    def format_tools_as_json(self):
        """Formats all tool schemas into a readable string format."""
        out = ""

        for tool in self.tools:
            tool_schema = tool.tool_json
            tool_name = tool_schema["name"]
            tool_description = tool_schema["description"]
            tool_arguments = tool_schema[
                "arguments"
            ]  # Changed from parameters to arguments

            out += f"""
            - **{tool_name}**: {tool_description}
            Arguments:
            {json.dumps(tool_arguments, indent=4)}
            """
        return out

    def _format_history_template(self, chat_history):
        """Formats the chat history section of the prompt."""
        return f"""
        **Current Message History:**
        {json.dumps(chat_history, indent=2)}
        """

    def _format_output_template(self):
        """Formats the output schema section with required JSON structure."""
        content_schema = (
            f"<JSON object defined by Pydantic JSON schema: {self.structured_output.model_json_schema()}>"
            if hasattr(self.structured_output, "model_json_schema")
            else (
                f'"{self.structured_output}"'
                if self.structured_output
                else '"<Your summarizing response or reasoning in Markdown format>"'
            )
        )

        return f"""
        Your responses MUST be written as a valid JSON object in the following structure:
        {{
            "role": "AI",
            "reasoning": "<list your reasoning about your "content" and your "tool_calls" here concisely>",
            "content": {content_schema},
            "tool_calls": [
                {{
                    "name": "<tool_name>",
                    "arguments": {{
                        "arg1": <LITERAL_VALUE>,
                        "arg2": <LITERAL_VALUE>
                    }}
                }}
            ] OR <null if no tool calls needed or no tools are available.>,
        }}

        Violating these requirements will cause errors. Fix by adjusting the response format.
        """

    def invoke(
        self, query, callbacks: Optional[CallbackManager] = None, **kwargs: Any
    ) -> Dict:
        """Executes the prompt template generation."""
        if callbacks:
            callbacks.on_invoke_start(self.name, query, **kwargs)
        result = self.update(query)
        if callbacks:
            callbacks.on_invoke_end(self.name, result, **kwargs)
        return result
