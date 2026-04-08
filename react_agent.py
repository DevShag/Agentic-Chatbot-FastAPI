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

#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
load_dotenv()

# Step 2 Setup LLM & Tools
openai_llm = ChatOpenAI(model='gpt-4o-mini')
groq_llm = ChatGroq(model='llama-3.3-70b-versatile')

search_tool = TavilySearchResults(max_results=2)

@tool
def calcualte(expression: str) -> str:
    """Evaluate a mathematical expression Exampl: '(10+5)*3'"""
    allowed = set("0123456789+-*/().**%")
    if not all (c in allowed for c in expression):
        return "Error: disallowed characters in expression."
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error: {exc}"
    

@tool
def get_current_time() -> str:
    """Return the current UTC date and time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("UTC: %Y-%m-%d %H:%M:%S")


TOOLS = [search_tool, calcualte, get_current_time]
    

class AgentState(TypedDict):
    """Shared state passed between every node in the graph"""
    messages : Annotated[list, add_messages]




# Step3: Setup AI Agent with Search tool functionality
def build_agent()->StateGraph:
    """
    Compile and return a ReAct agent as a LangGraph StateGraph.
 
    Graph topology
    ──────────────
    START ──► agent ──► (tool_calls?) ──YES──► tools ──► agent
                                      └─NO──► END
    """

    llm_with_tools = openai_llm.bind_tools(TOOLS)

    SYSSTEM_PROMPT = SystemMessage(content=(
        "You are a helpful AI assistant with access to tools. "
        "Use them whenever they help you give a better answer."
        "Think step-by-step before acting."
    ))


    #--------------- Node 1: Agent Node calls the  -------------------
    def agent_node(state: AgentState) -> AgentState:
        message = [SYSSTEM_PROMPT] + state["messages"]
        response = llm_with_tools.invoke(message)
        return {'messages' : [response]}
    

    #--------------- Node 2: run whatever tools the LLM requestd ---------------
    tool_node = ToolNode(TOOLS)


    #-------------- Control Flow (ReAct loop) -----------------------
    def should_continue(state: AgentState) -> str:
        last = state['messages'][-1]
        return 'tools' if last.tool_calls else END
    

    #-------------- Assemble ---------------------------------
    graph = StateGraph(AgentState)

    graph.add_node('agent_node', agent_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'agent_node')
    graph.add_conditional_edges('agent_node', should_continue)
    graph.add_edge('tools', 'agent_node') # after tool -> back to agent 

    return graph.compile()


#--------------------------------------------------------------
# 4. Helpers
#--------------------------------------------------------------

def ask(agent, question: str) -> str:
    """
    Stream a single question through the agent, printing each step.
    """
    print(f"\n{'-'*60}")
    print(f"? {question}")
    print("-" * 60)
    last_chunk = None
    for chunk in agent.stream(
        {'messages': [HumanMessage(content=question)]},
        stream_mode = 'values',
    ):
        chunk['messages'][-1].pretty_print()
        last_chunk = chunk
    return last_chunk['messages'][-1].content



#-------------------------------------------------------------------------
# 5. Demo
#-------------------------------------------------------------------------

if __name__ == "__main__":
    agent = build_agent()

    ask(agent, "Calculate (256 * 3) + (1000 / 4) and define what LLM means.")









