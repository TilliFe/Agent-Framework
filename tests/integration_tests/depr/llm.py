from mini import LLMModel, Chat, tool, add_numbers, subtract_numbers


def test_llm():

    # Define the model
    model = Chat(LLMModel())

    # Bind tools to the model and enforce the use of the add_numbers tool
    model.bind_tools(
        tools=[tool(add_numbers), tool(subtract_numbers)], tool_choices=["add_numbers"]
    )

    # Define the structured output
    strucutred_output = {"computation": "<python expression>", "result": "<int>"}
    model.with_structured_output(strucutred_output)

    # Call the model with a query
    response = model.invoke("What is 6 plus 7?")
