import streamlit as st
import os
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_agent_executor
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage 
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
import operator
import json 
from dotenv import load_dotenv
from tools import e2b_data_analysis_tool, SearchDocuments
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from tools import web_search_tool


class Code(BaseModel):
    """Plan to follow in future"""

    prefix: str = Field(description='Detailed description of the problem statement and the python implementations assumptions.')
    imports: str = Field(
        description="detailed python code block containing only imports required for the `code` block to execute"
    )
    code: str = Field(
        description="Detailed optmized error-free single python function WITHOUT imports."
    )
    tests: Optional[str] = Field(
        description='Detailed Functional tests for the generated code'
    )


class AgentCoder(TypedDict):
    requirement: str
    supporting_docs: Optional[str]
    agent_scratchpad: Optional[str]
    code: Code
    errors: Optional[str]
    iterations: int
    metadata: Optional[str]