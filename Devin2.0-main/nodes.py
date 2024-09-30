from datamodel import AgentCoder
from tools import SearchDocuments

import json 
from tools import e2b_data_analysis_tool, SearchDocuments
from datamodel import Code
from agents import coder, qa_tester, refiner, researcher, pm, architect


def explorer(state: AgentCoder):
    print('Entering in Explorer')
    requirement = state['requirement']
    iterations = state['iterations']
    thoughts = state['agent_scratchpad']
    response = researcher.invoke({'requirement': requirement, 'agent_scratchpad':thoughts})
    output = response['output']
    scratchpad = response['agent_scratchpad']


    return {'requirement': requirement, "agent_scratchpad":scratchpad,'supporting_docs': output , 'iterations': iterations, 'metadata': state['metadata']}

def planner(state: AgentCoder):
    print('Entering in Planner')
    requirement = state['requirement']
    iterations = state['iterations']
    supporting_docs = state['supporting_docs']

    response = pm.invoke({'requirement': requirement, 'supporting_docs': supporting_docs })
    output = response['output']

    return {'requirement': requirement, 'supporting_docs': output, 'iterations': iterations, 'metadata': state['metadata']}

def srengineer(state: AgentCoder):
    print('Entering in Sr. Engineer Architect')
    requirement = state['requirement']
    iterations = state['iterations']
    supporting_docs = state['supporting_docs']
    metadata = state['metadata']

    response = architect.invoke({'requirement': requirement, 'plan_of_execution': supporting_docs, 'metadata': metadata })
    output = response['output']
    
    return {'requirement': requirement, 'iterations': iterations, 'metadata': metadata}

    

def programmer(state: AgentCoder):
    print(f'Entering in Programmer')
    requirement = state['requirement']
    iterations = state['iterations']
    print('-----------Fetching Relevant Docs-------------')
    docs = SearchDocuments().run(requirement)
    code_solution: Code = coder.invoke({'requirement':requirement, 'docs': docs})
    code_solution.imports = code_solution.imports or 'import math'
    iterations += 1
    return {'code':code_solution, 'iterations': iterations, 'requirement': requirement}


def tester(state: AgentCoder):
    print(f'Entering in Tester')
    requirement = state['requirement']
    iterations = state['iterations']
    code_solution: Code = state['code'] 
    errors = state['errors']

    code_from_tester = qa_tester.invoke({'requirement':requirement,'code':code_solution.code})
    code_solution.tests = code_from_tester.tests
    
    return AgentCoder(**{'code':code_solution, 'iterations': iterations, 'requirement': requirement, 'errors': errors} )

def executer(state: AgentCoder):
    print("---CHECKING CODE---")
    code_solution = state["code"]
    iterations = state["iterations"]
    requirement = state["requirement"]

    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code
    tests = code_solution.tests

    print(imports, prefix)
    # Check imports
    try: 
        e2b_ouput = json.loads(e2b_data_analysis_tool.run(imports) ) 
    except IndexError:
        e2b_ouput = {'stderr' : '', 'stdout': ''}

    print(e2b_ouput)
    err = e2b_ouput['stderr']
    if err: 
        print("---CODE IMPORT CHECK: FAILED---")
        return AgentCoder( **{"code": code_solution, "errors": 'Your code failed while doing imports with the following error: '  + err, "iterations": iterations, 'requirement': requirement} ) 
    
    # Check execution
    e2b_ouput = json.loads( e2b_data_analysis_tool.run(f"{imports}\n{code}") ) 
    err = e2b_ouput['stderr']
    if err: 
        return AgentCoder( **{"errors": 'Your code failed while executing the code blocks with the following error: ' + err, "code": code_solution, "iterations": iterations, 'requirement': requirement} ) 
  
    # Check execution
    e2b_ouput = json.loads( e2b_data_analysis_tool.run(f"{imports}\n{code}\n{tests}") ) 
    err = e2b_ouput['stderr']
    if err: 
        return AgentCoder( **{"errors": 'Your code failed while executing the tests with the following error: ' + err, "code": code_solution, "iterations": iterations, 'requirement': requirement} ) 

    print("---NO CODE TEST FAILURES---")
    return AgentCoder( **{"code": code_solution, "iterations": iterations, "errors": None, 'requirement': requirement} )

def debugger(state):
    print(f'----Entering in Debugger----')
    errors = state['errors']
    code_sol = state['code']
    iterations = state['iterations']
    refine_code_ = refiner.invoke({'code':code_sol,'errors':errors})
    return AgentCoder( **{"code":refine_code_,"errors":'', "iterations" : iterations  })

def decide_to_end(state):
    print(f'Entering in Decide to End')
    iterations = state['iterations']
    if state['errors'] and iterations < 3:
        return 'debugger'
    else:
        return 'end'