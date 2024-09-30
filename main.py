from streamlit_monaco import st_monaco
import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
import asyncio
from barfi import st_barfi, Block, barfi_schemas
from io import StringIO
import pandas as pd
import tabulate 
import langchain

import json 
from tools import e2b_data_analysis_tool
from agents import llm
from datamodel import AgentCoder
from nodes import programmer, debugger, tester, executer, decide_to_end, explorer, srengineer, planner
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from utils import get_dataframe_metadata

load_dotenv(".env.var")

workflow = StateGraph(AgentCoder)

# Build graph
workflow.add_node("explorer", explorer) 
workflow.add_node("planner", planner) 
workflow.add_node("architect", srengineer) 
workflow.add_node("debugger", debugger) 
workflow.add_node("tester", tester)
workflow.add_node("executer", executer)
workflow.add_node("programmer", programmer)

workflow.set_entry_point("explorer")
workflow.add_edge("explorer", "planner")
workflow.add_edge("planner", "architect")
workflow.add_edge("architect", "programmer")
workflow.add_edge("programmer", "tester")
workflow.add_edge("tester", "executer")
workflow.add_edge("debugger", "executer")

workflow.add_conditional_edges(
    "executer",
    decide_to_end,
    {
        "end": END,
        "debugger": "debugger",
    },
)

app = workflow.compile()

async def main(inputs) :
    config = {"recursion_limit": 50}
    async for event in app.astream(inputs, config=config):
        for node, state in event.items():
            if node != "__end__":
                print(state)
                    
                with st.status(node):
                    if 'errors' in state and state['errors']: 
                        st.error(state['errors'])
                    with st.chat_message('Bot'):
                        if 'code' in state: 
                            full_code = f"""{state['code'].imports}\n{state['code'].code}\n\n{state['code'].tests} """
                            st.markdown(f''' ```py {full_code}``` ''')
                            st.session_state.messages.append({"role": "Bot", "content": f''' ```py {full_code}``` '''})
                            st.session_state.code = full_code
                        elif 'supporting_docs' in state: 
                            st.session_state.supporting_docs = state['supporting_docs']
                            st.markdown(st.session_state.supporting_docs)
                        else:
                            st.markdown("Success!")
                print('----------'*20)

                
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

                
if __name__ == '__main__':

    running_dict = {}


    loop = get_or_create_eventloop()

    st.set_page_config(layout='wide')

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {'role': 'assistant', 'content': 'Hello! Lets get started on creating your workflow. What would you like to do today?'}
        ]

    if "code" not in st.session_state:
        st.session_state.code = 'Generating code ...' 
    if "supporting_docs" not in st.session_state:
        st.session_state.supporting_docs = 'Thinking...'
    

    


    with st.sidebar:
        data = df = pd.DataFrame({
                'Column1': [1, 2, 3, 4],
                'Column2': ['A', 'B', 'C', 'D'],
                'Column3': [0.1, 0.2, 0.3, 0.4]})

        messages = st.container()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, type=['csv'])
        for uploaded_file in uploaded_files:
            @st.cache_data 
            def load_data(url):
                df = pd.read_csv(url)
                return df
            data = load_data(uploaded_file)
        if prompt := st.chat_input("Type your prompt..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "Bot", "content": "Assigning best agents on that task..."})
            with st.chat_message("Bot"):
                st.markdown('Assigning best agents on that task...')

            inputs = {"requirement": prompt, 'iterations': 0, "metadata": get_dataframe_metadata(data)}
            loop.run_until_complete(main(inputs))

    
    tab1, tab2, tab3, tab4 = st.tabs(["📆 Planner","💻 Code", "📈 Graph", "📋 Table"])
    feed = Block(name='Feed' )
    feed.add_output()

    result = Block(name='Result')
    result.add_output()
    blocks = [feed, result]

    with tab1:
        with st.container():
            editor_planner = st.markdown(st.session_state.supporting_docs)

    with tab2: 
        with st.container(height= 700):
            editor = st_monaco(value= st.session_state.code , language='python', minimap=True, theme='white', height=700 )
    with tab3: 
        with st.container():
            st_barfi(base_blocks=blocks)
        
    with tab4:
        with st.container():
            st.data_editor(data)
            dataframe_agent = create_pandas_dataframe_agent(llm=llm, df=data, agent_type="openai-tools")
            if data_prompt := st.chat_input("Ask about your dataset..."):
                with st.chat_message("user"):
                    st.markdown(data_prompt)
                with st.chat_message("ai"):
                    st.markdown('Assigning best agents on that task...')
                st.write(dataframe_agent(data_prompt)["output"])



            

   
