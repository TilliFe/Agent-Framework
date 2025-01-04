import json
from pydantic import ValidationError
from typing import Optional, Any
from .prompt_template import PromptTemplate
from .utils import extract_json, HumanMessage, ToolMessage, ToolResponse
from .runnable import Runnable, CallbackManager


class AgentExecutor(Runnable):
    """Executes agent interactions with LLM models, managing tools, system prompts, and structured outputs."""

    def __init__(
        self,
        llm_model,
        system=None,
        tools=None,
        tool_choices=None,
        messages=None,
        structured_output=None,
        max_retries=5,
    ):
        self.llm_model = llm_model
        self.system = system
        self.tools = tools
        self.tool_choices = tool_choices
        self.temp_called_tools = []
        self.structured_output = structured_output
        self.messages = messages
        self.max_retries = max_retries
        self.template = ""
        self.update_template()

    def update_template(self):
        """Updates the prompt template with current configuration."""
        self.template = PromptTemplate(
            system=self.system,
            tools=self.tools,
            tool_choices=self.tool_choices,
            structured_output=self.structured_output,
        )

    def bind_tools(self, tools, tool_choices=None):
        """Assigns new tools to the agent."""
        self.tools = tools
        self.tool_choices = tool_choices
        self.update_template()

    def reset_tools(self):
        """Removes all tools from the agent."""
        self.tools = []
        self.update_template()

    def bind_system(self, system):
        """Sets a new system prompt."""
        self.system = system
        self.update_template()

    def reset_system(self):
        """Clears the system prompt."""
        self.system = None
        self.update_template()

    def with_structured_output(self, structured_output):
        """Sets the structured output format."""
        self.structured_output = structured_output
        self.update_template()

    def reset_structured_output(self):
        """Removes the structured output configuration."""
        self.structured_output = None
        self.update_template()

    def reset_template(self):
        """Resets all template configurations to default state."""
        self.system = None
        self.tools = []
        self.structured_output = None
        self.update_template()

    def invoke(
        self,
        query,
        print_intermediate_steps=False,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        """Processes a query through the agent, handling tools and structured outputs."""
        if callbacks:
            callbacks.on_invoke_start(self.name, query, **kwargs)

        if print_intermediate_steps:
            print("\033[90m\033[1mQuery:\033[0m\033[90m " + query + "\033[0m")
        try:
            # Call the model for the first time with the humans query.
            chat_history = self.messages if self.messages is not None else []
            chat_history.append(HumanMessage(query)())
            prompt = self.template.update(chat_history)
            parsed_response = extract_json(self.llm_model.invoke(prompt))
            chat_history.append(parsed_response)

            if print_intermediate_steps:
                (
                    print(
                        "\033[90m\033[1mAI:\033[0m\033[90m "
                        + (
                            json.dumps(chat_history[-1]["content"])
                            if isinstance(chat_history[-1]["content"], dict)
                            else chat_history[-1]["content"]
                        )
                        + "\033[0m"
                    )
                    if chat_history[-1]["content"]
                    else ""
                )

            parsed_response = self._handle_tool_calls(
                parsed_response,
                chat_history,
                print_intermediate_steps=print_intermediate_steps,
            )

            # check for type correctness, but only if the structured output is a Pydantic model
            if self.template.structured_output and hasattr(
                self.template.structured_output, "model_json_schema"
            ):

                num_tries = 1
                max_tries = 3

                validated, error = self.is_valid_json(
                    parsed_response["content"], self.template.structured_output
                )

                while not validated and num_tries <= max_tries:
                    num_tries += 1
                    chat_history.append(
                        HumanMessage(
                            f"""The validation with the Pydantic Class did not return True. Here is the Pydantic error message: {error}. Don't be apologetic! Provide the correct message with the pydantic "content" field directly, nothing else."""
                        )()
                    )
                    prompt = self.template.update(chat_history)
                    parsed_response = extract_json(self.llm_model.invoke(prompt))
                    chat_history.append(parsed_response)

                    validated, error = self.is_valid_json(
                        parsed_response["content"], self.template.structured_output
                    )

                accepted_message = parsed_response

                if num_tries > 1:
                    # remove all other messages from chat history about the structured output
                    for _ in range(num_tries * 2 + 1):
                        chat_history.pop()

                    chat_history.append(accepted_message)

            # Update the Chat's messages
            if self.messages is not None:
                self.messages = chat_history

            self.temp_called_tools = []

            if callbacks:
                callbacks.on_invoke_end(self.name, parsed_response, **kwargs)

            return parsed_response["content"]

        except Exception as e:
            return {f"Error: {str(e)}"}

    def is_valid_json(self, json_data, model_class):
        """Validates JSON data against a Pydantic model."""
        try:
            _ = model_class.model_validate(json_data)
            return True, None
        except ValidationError as e:
            return False, e
        except Exception as e:
            return False, e

    def _handle_tool_calls(
        self, parsed_response, chat_history, depth=0, print_intermediate_steps=False
    ):
        """Manages the execution flow of tool calls and their responses."""
        if depth > 3:
            return parsed_response

        # check if all tools in self.tool_are called at the end
        while parsed_response.get("tool_calls"):
            tool_usage_summary = parsed_response["content"]
            tool_response = self.execute_tool_calls(
                parsed_response, print_intermediate_steps=print_intermediate_steps
            )
            chat_history.append(tool_response)

            tool_calling_info = ""
            tool_message = chat_history[-2]
            for tool in tool_message["tool_calls"]:
                tool_calling_info += f"{tool['name']} "
                if self.tool_choices:
                    self.temp_called_tools.append(tool["name"])

            prompt = self.template.update(chat_history)
            parsed_response = extract_json(self.llm_model.invoke(prompt))
            chat_history.append(parsed_response)
            if print_intermediate_steps:
                (
                    print(
                        "\033[90m\033[1mAI:\033[0m\033[90m "
                        + (
                            json.dumps(chat_history[-1]["content"])
                            if isinstance(chat_history[-1]["content"], dict)
                            else chat_history[-1]["content"]
                        )
                        + "\033[0m"
                    )
                    if chat_history[-1]["content"]
                    else ""
                )

        # check if all tools are called at the end, compare against the self.tool_choices lsit which holds the names of the tools
        if self.tool_choices and set(self.temp_called_tools) != set(self.tool_choices):
            not_used_tools = list(set(self.tool_choices) - set(self.temp_called_tools))
            chat_history.append(
                HumanMessage(
                    f"""The tools {not_used_tools} were not used in the conversation."""
                )()
            )
            prompt = self.template.update(chat_history)
            parsed_response = extract_json(self.llm_model.invoke(prompt))
            chat_history.append(parsed_response)

            self._handle_tool_calls(parsed_response, chat_history, depth=depth + 1)

        return parsed_response

    def execute_tool_calls(self, parsed_responses, print_intermediate_steps=False):
        """Executes the requested tools and returns their responses."""
        parsed_responses = parsed_responses["tool_calls"]

        tool_names = [tool.tool_json["name"] for tool in self.tools]

        response_list = [] if type(parsed_responses) == list else [parsed_responses]

        for parsed_response in parsed_responses:
            function_name = parsed_response["name"]
            function_args = parsed_response["arguments"]

            if function_name in tool_names:
                function_index = tool_names.index(function_name)
                function_to_call = self.tools[function_index].executable

                (
                    print(
                        "\033[90m\033[1mCalling Tool:\033[0m\033[90m "
                        + function_name
                        + "\033[0m"
                    )
                    if print_intermediate_steps
                    else ""
                )

                try:
                    result = function_to_call(**function_args)
                    response_list.append(ToolResponse(function_name, str(result))())
                    if print_intermediate_steps:
                        print(
                            "\033[90m\033[1mTool Result:\033[0m\033[90m "
                            + str(result)
                            + "\033[0m"
                        )
                except (TypeError, ValueError) as e:
                    response_list.append(
                        ToolResponse(function_name, f"Error: {str(e)}")()
                    )

            else:
                response_list.append(
                    ToolResponse(
                        function_name,
                        "ERROR: Unknown tool. Try another tool or don't use a tool.",
                    )()
                )

        return ToolMessage(response_list)()
