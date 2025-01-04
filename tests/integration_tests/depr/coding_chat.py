from framework import (
    Chat,
    LLMModel,
    tool,
    execute_python_code,
)
import json
from pydantic import BaseModel, Field

# CODE GENERATOR - Focuses purely on code generation
code_generator_role = """
    Expert Python code generator that:
    1. Writes clean, PEP 8 compliant implementations
    2. Includes docstrings and type hints
    3. Creates modular, reusable code
    Focus only on core implementation - no tests or review comments.
"""


class CodeGeneratorOutput(BaseModel):
    code: str = Field(..., description="Clean, executable Python implementation")


code_generator = Chat(
    llm_model=LLMModel(),
    system=code_generator_role,
    messages=[],
    structured_output=CodeGeneratorOutput,
)

# CODE REVIEWER - Focuses on targeted feedback
code_reviewer_role = """
    Expert code reviewer focused on:
    1. Code correctness and reliability
    2. Performance and security issues  
    3. Design anti-patterns
    Provide specific line-by-line feedback only, no rewrites. Don't be nice!
"""


class CodeReviewerOutput(BaseModel):
    issues: str = Field(..., description="Specific issues found in the code")
    needs_improvement: bool = Field(..., description="Whether code requires iteration")


code_reviewer = Chat(
    llm_model=LLMModel(),
    system=code_reviewer_role,
    structured_output=CodeReviewerOutput,
)

# CODE TESTER - Focuses purely on testing
code_tester_role = """
    Testing specialist that:
    1. Writes compact unit tests including edge cases
    2. Executes via execute_python_code tool
    3. Reports pass/fail and stops on errors
    4. Iterates if tests fail, blaming code issues

    Always wrap full test code with implementation and run with tool.
"""


class TestResult(BaseModel):
    passed: bool = Field(..., description="True if all tests passed, False otherwise")


code_tester = Chat(
    llm_model=LLMModel(),
    system=code_tester_role,
    tools=[tool(execute_python_code)],
    structured_output=TestResult,
    messages=[],
)


def test_coding_chat(num_iterations=2):
    """Test the multi-chat system for generating, reviewing, and testing code."""

    # Define the query
    query = "How do I write a DFS search from scratch?"

    # Initialize variables
    generated_code = ""
    review = query
    needs_improvement = True

    # Iterate until code is reviewed and tested successfully
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}:")

        generated_code = code_generator.invoke(review)["code"]
        review = code_reviewer.invoke(
            f"""Based on this query: {query}, I got this code: {generated_code}"""
        )
        needs_improvement = review["needs_improvement"]
        review = json.dumps(review)

        if not needs_improvement:
            break

        # tests_passed = code_tester.invoke(generated_code)

    # Final output
    print(generated_code)
