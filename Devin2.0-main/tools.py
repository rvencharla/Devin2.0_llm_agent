
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.tools import BearlyInterpreterTool, E2BDataAnalysisTool
from langchain.tools import BaseTool
from langchain.agents import AgentType, Tool, initialize_agent

import settings
import os 

from typing import Any, Coroutine
import requests


os.environ['SEARCHAPI_API_KEY'] = settings.SEARCHAPI_API_KEY
os.environ["E2B_API_KEY"] =  settings.E2B_API_KEY # upto 100$ free code execution 

search = SearchApiAPIWrapper()
bearly_tool = BearlyInterpreterTool(api_key= settings.BEARLY_API_KEY) # code execution upto 100 requests 

e2b_data_analysis_tool = E2BDataAnalysisTool(
    on_stdout=lambda stdout: print("stdout:", stdout),
    on_stderr=lambda stderr: print("stderr:", stderr),
)


web_search_tool =  Tool(
        name="search_web",
        func=search.run,
        description="Useful when unsure about information that might change often like news, markets, code documentations, fashion and time-sensitive information.",
    )


class SearchDocuments(BaseTool):
    name ='SearchDocuments'
    description = 'Used to search the database of documentations for scientific libraries if there are docs relevant to the given query'

    def _run(self, query: str) -> str:
        """
            Parameters:
            - query (str): The search string to query the database. This should be a simple
                        text string that describes what context is needed.
            Returns:
            - str: A text string encoded version of list of dictionaries, containing the context fetched from the database. 
        """
        
        encoded_query = requests.utils.quote(query)
        url = f"https://flask-production-751b.up.railway.app/getTopContexts?course_name=workflow_generator&search_query={encoded_query}&token_limit=3000"

        response = requests.request("GET", url)

        data = response.json()
        formatter = lambda  l: '---\n' + '---\n'.join( 
            f'''{str(i + 1)}: {d['readable_filename']}, page: {d.get('pagenumber', '')}\n {d['text']}\n''' 
            for i, d in enumerate(l)
        )

        return formatter(data)
    

    
class SearchWolframAlpha(BaseTool):
    name ='SearchWolframAlpha'
    description = 'Used to search the expert on science, math, engineering and any computational knowledge.'

    def _run(self, query: str) -> str:
        """
            Parameters:
            - query (str): The search string to query the WolframAlpha expert DB. This should be a simple
                        text string that describes what question needs to be answered in math, science, engineering and any computational knowledge. 
            Returns:
            - str: A text string encoded version of an answer to given search query generated by an expert. 
        """
        api_endpoint = "https://www.wolframalpha.com/api/v1/llm-api"
        api_key = settings.WOLFRAM_LLM_API_KEY

        response = requests.get(
            api_endpoint,
            params={
                "i": query,
                "appid": api_key
            }
        )
        print(response.status_code)
        return response.content
        
     
