import os
from typing import Dict, TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from agents.prompts import RESEARCH_PROMPT, ANALYST_PROMPT, DECISION_PROMPT, CRITIC_PROMPT

load_dotenv()

# Lazy initialization — avoids crash at import time if packages are missing
_llm = None
_search_tool = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    return _llm

def get_search_tool():
    global _search_tool
    if _search_tool is None:
        from langchain_community.tools import DuckDuckGoSearchResults
        _search_tool = DuckDuckGoSearchResults()
    return _search_tool

class AgentState(TypedDict):
    query: str
    research_output: str
    analysis_output: str
    decision_output: str
    critic_output: str
    iterations: int

def research_node(state: AgentState):
    query = state["query"]
    # 1. Search DuckDuckGo (facts only)
    try:
        search_results = get_search_tool().invoke(query)
    except Exception as e:
        search_results = f"Search failed: {e}"

    prompt = RESEARCH_PROMPT.format(input=f"Query: {query}\nSearch Results: {search_results}")
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"research_output": response.content}

def analyst_node(state: AgentState):
    prompt = ANALYST_PROMPT.format(research_output=state["research_output"])
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"analysis_output": response.content}

def decision_node(state: AgentState):
    prompt = DECISION_PROMPT.format(analysis_output=state["analysis_output"])
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"decision_output": response.content}

def critic_node(state: AgentState):
    prompt = CRITIC_PROMPT.format(decision_output=state["decision_output"])
    response = get_llm().invoke([HumanMessage(content=prompt)])
    iterations = state.get("iterations", 0) + 1
    return {"critic_output": response.content, "iterations": iterations}

# Build the orchestration graph once at module load (no heavy objects created here)
_workflow = StateGraph(AgentState)
_workflow.add_node("research", research_node)
_workflow.add_node("analyst", analyst_node)
_workflow.add_node("decision", decision_node)
_workflow.add_node("critic", critic_node)
_workflow.set_entry_point("research")
_workflow.add_edge("research", "analyst")
_workflow.add_edge("analyst", "decision")
_workflow.add_edge("decision", "critic")
_workflow.add_edge("critic", END)

app_graph = _workflow.compile()

def run_decision_system(query: str) -> Dict[str, Any]:
    initial_state = {
        "query": query,
        "research_output": "",
        "analysis_output": "",
        "decision_output": "",
        "critic_output": "",
        "iterations": 0
    }
    result = app_graph.invoke(initial_state)
    return result

