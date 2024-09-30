from datamodel import Code 
import streamlit as st
import os
from langchain_openai import AzureChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
import settings
import numexpr
from tools import SearchWolframAlpha, SearchDocuments, web_search_tool
from langgraph.prebuilt import ToolExecutor
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import initialize_agent, AgentType
from langgraph.prebuilt import chat_agent_executor
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools, load_agent
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

llm = AzureChatOpenAI(
    openai_api_version= settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_deployment = settings.AZURE_OPENAI_MODEL,
    temperature=0.2
)

def create_explorer(): 
    tools = [ SearchWolframAlpha() , SearchDocuments(), web_search_tool]


    explorer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''**Role**: 
                You're a world-class researcher and python expert. Your only task is to gain more information about a given REQUIREMENT.
                Use all the available tools to search and retrieve anything relevant to writing a python code workflow for the given task. 
                Don't give suggestions or opinions, just give facts and gathered information. 
                Finally, give a very organized and detailed document of all collected information that a different planner can use to make a detailed plan. 
                
                **Instructions**:
                1. **Understand and Clarify**: Make sure you understand the task down to its details.
                2. **Search**: Look up different ways to approach the problem. Gather adequate alternative approaches and competing information. 
                3. **Organized**: Output an organized document of similar and dissimilar information. 

                '''
            ),
            ("human", "{requirement}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
  
    explorer = create_tool_calling_agent(llm, tools, explorer_prompt)
    # agent = (
    #     {
    #         "agent_scratchpad": lambda x: format_to_openai_function_messages(),
    #     }
    #     | explorer
    #     | OpenAIFunctionsAgentOutputParser()
    # )

    return AgentExecutor(agent=explorer, tools=tools, verbose=True, return_intermediate_steps=True)


    
def create_planner(): 
    tools = [SearchWolframAlpha(), SearchDocuments(), web_search_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''**Role**: 
                You're a world-class Engineering Manager. Your only task is make a VERY detailed plan of execution for the given REQUIREMENT.
                Make very informed decision for EACH step baesd on the given SUPPORTING INFORMATION. 
                Use all the available tools to search and retrieve anything information justifying each step of your plan. 
                Finally, give a very organized and detailed execution strategy for the given task based on the prompt. 
                
                **Instructions**:
                1. **Understand and Clarify**: Make sure you understand each step of execution down to its details.
                2. **Search**: Look up the pros and cons of the choice you make for each step.
                3. **Organized**: Output an extremely organized and step-by-step execution plan for the given requirement. 
                4. **Output**: For each step in the final output, it SHOULD contain function name with type annotations and example of what the input and output is.
                '''
            ),
            ("human", "*SUPPORTING INFORMATION*: {supporting_docs}\n *REQUIREMENT*: {requirement} "),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
  
    planner = create_tool_calling_agent(llm, tools, prompt)


    return AgentExecutor(agent=planner, tools=tools, verbose=True)

 
def create_architect(): 
    tools = [  SearchDocuments(), web_search_tool ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''**Role**: 
                You're a world-class Senior Software Architect. Given a plan of execution document, your only task is make a VERY detailed skeleton python code for the given REQUIREMENT.
                A skeleton python code has function names, input and ouput parameters with type annotations and copious amounts of docstrings.
                The skeleton code also has order of execution and transforms the output from one function into another. Leave the inner implementational details out.
                Use functional style of coding with very little side-effects. 

                The very first function's input is a CSV of the given structure. As part of docstrings, add the input csv structure and output csv structure for EACH FUNCTION.

                Make very thorough and fine-grained functions for each step with copious documentations and implementational suggestions as comments.
                Use all the available tools to search and retrieve documentations for python libraries and functions that might be used. 
                Finally, output a single Python file with skeleton python code that satisfies EVERY step in the plan of execution strategy document. 
                
                **Instructions**:
                1. **Understand and Clarify**: Make sure you understand each step of execution down to its details.
                2. **Search**: Use the tools to look up the documentations for libraries and other implementational requirements for each step in skeleton code.
                3. **Organized**: Output an extremely organized and step-by-step python code with functions. 
                4. **Output**: For each step in python skeleton code, it SHOULD contain function name with type annotations and example of what the input and output is along with implementational details.

                **CSV STRUCTURE**:
                {metadata}
                '''
            ),
            ("human", "*PLAN OF EXECUTION STRATEGY*: {plan_of_execution}\n *REQUIREMENT*: {requirement} "),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
  
    architect = create_tool_calling_agent(llm, tools, prompt)


    return AgentExecutor(agent=architect, tools=tools, verbose=True)
     

def create_programmer():
    code_gen_prompt = ChatPromptTemplate.from_template(
        '''**Role**: You are a expert software python programmer. You need to develop a single python function that closely and accurately solves the User Task. 
    **Task**: As a programmer, you are required to complete the function. 
    Think through the problem step-by-step to break down the problem, then write the code in Python language. 
    Ensure that your code is efficient, readable, and well-commented.
    Write a SINGLE self-contained function and its imports. NO variables in global scope. Use functional programming style. 
    If there is a missing module error, use python's `os.shell()` to install all the required packages at the top of the file. 
    Write docstrings that clearly state what the expected input and output format is with examples. 

    **Instructions**:
    1. **Understand and Clarify**: Make sure you understand the task.
    2. **Search**: Look up any required documentations or other required libraries that you might need for this Task. 
    3. **Algorithm/Method Selection**: Decide on the most efficient way. 
    4. **Code Generation**: Translate your pseudocode into executable Python code
    5. **Use Docs**: Use the attached documentations and code snippets if relevant to solve the task.

    *DOCUMENTATION*:
    {docs}
    *REQURIEMENT*
    {requirement}'''
    )
    coder = create_structured_output_runnable(
        Code, llm , code_gen_prompt, mode='openai-tools'
    )
    return coder 


def create_debugger():
    python_refine_gen = ChatPromptTemplate.from_template(
        """
        You are an expert AI coder who makes no mistake. You tried to solve this problem and failed due to the given error. 
        Reflect on this failure given the provided documentation. Think of a few key suggestions based on the documentation to avoid making this mistake again.
        Finally, return the final modified code free of errors and that solves the given error below in the EXACT structure requested. 
      
        *Code*: {code}
        *Errors*: {errors}
        """
    )
    refine_code = create_structured_output_runnable(
        Code, llm, python_refine_gen
    )
    return refine_code


def create_tester() : 

    
    test_gen_prompt = ChatPromptTemplate.from_template(
        '''
    **Role**: You're an expert tester, your task is to create Python test-cases based on provided Requirement for the given Python Code. 
    Each test should call the given Python Code functions and assert an expected output. Write both positive and negative test-cases. 

    Your job depends on this! Be accurate and cover all possible test-cases. 
    These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
    **1. Basic Test Cases**:
    - **Objective**: Basic and Small scale test cases to validate basic functioning 
    **2. Edge Test Cases**:
    - **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

    **Instructions**:
    - Implement a comprehensive set of test cases based on requirements.
    - Pay special attention to edge cases as they often reveal hidden bugs.
    - Only Generate Basics and Edge cases which are small
    - Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases

    **REQURIEMENT**:
    {requirement}

    **Code**:
    {code}
    '''
    )
    tester_agent = create_structured_output_runnable(
        Code, llm, test_gen_prompt
    )
    return tester_agent



coder = create_programmer()
refiner = create_debugger()
qa_tester = create_tester()
researcher = create_explorer()
pm = create_planner()
architect = create_architect()