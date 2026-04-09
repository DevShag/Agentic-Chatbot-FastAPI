# 1. Setup Imports
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


load_dotenv()

system_prompt="Act as an AI chatbot who is smart and friendly"

class ChatState(TypedDict):
    """Shared state passed between every node in the graph"""
    messages : Annotated[list, add_messages]



def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=='Groq':
        llm = ChatGroq(model=llm_id)
    elif provider == 'OpenAI':
        llm = ChatOpenAI(model=llm_id)


    TOOLS = [TavilySearchResults(max_results=2)] if allow_search else[]
    agent = build_agent(llm, TOOLS, system_prompt)

    chat_state = {'messages': query}
    response = agent.invoke(chat_state)
    #chatbot.invoke(initial_state)['messages'][-1].content
    return response['messages'][-1].content
   


# Step3: Setup AI Agent with Search tool functionality
def build_agent(llm, TOOLS, system_prompt)->StateGraph:
    llm_with_tool = llm.bind_tools(TOOLS)


    #-------------------Node 1: Agent Node-------------------
    def agent_node(state: ChatState) -> ChatState:
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        #[system_prompt] + state['messages']
        response = llm_with_tool.invoke(messages)
        return {'messages': [response]}
    

    #--------------------Node 2: run whatever tools the LLM requested --------------
    tool_node = ToolNode(TOOLS)


    #-------------------- Control Flow (React loop) ---------------
    def should_continue(state: ChatState) -> str:
        last = state['messages'][-1]
        return 'tools' if last.tool_calls else END
    
    #---------------------- Assemble------------------------------
    graph = StateGraph(ChatState)

    graph.add_node('agent', agent_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', should_continue)
    graph.add_edge('tools', 'agent')

    return graph.compile()



