import os
import time
from typing import Dict, TypedDict, Any

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from agents.prompts import (
    RESEARCH_PROMPT,
    ANALYST_PROMPT,
    DECISION_PROMPT,
    CRITIC_PROMPT
)

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 3
INITIAL_BACKOFF = 2          # seconds
LLM_TIMEOUT = 30             # seconds


# Lazy initialization
_llm = None
_search_tool = None


def get_llm():
    global _llm

    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            timeout=LLM_TIMEOUT
        )

    return _llm


def get_search_tool():
    global _search_tool

    if _search_tool is None:
        from langchain_community.tools import DuckDuckGoSearchResults
        _search_tool = DuckDuckGoSearchResults()

    return _search_tool


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    query: str
    research_output: str
    analysis_output: str
    decision_output: str
    critic_output: str
    iterations: int


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_rate_limit_error(error: Exception) -> bool:
    """
    Detect whether an exception is likely caused by
    API rate limiting.
    """

    error_message = str(error).lower()

    rate_limit_keywords = [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded"
    ]

    return any(
        keyword in error_message
        for keyword in rate_limit_keywords
    )


# ============================================================
# SAFE LLM INVOCATION
# ============================================================

def safe_llm_invoke(prompt: str) -> str:
    """
    Calls the LLM with:
    - timeout
    - rate-limit detection
    - exponential backoff
    - limited retries
    - graceful failure
    """

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = get_llm().invoke(
                [HumanMessage(content=prompt)]
            )

            return response.content

        except Exception as e:

            # --------------------------------------------
            # RATE LIMIT
            # --------------------------------------------

            if is_rate_limit_error(e):

                if attempt < MAX_RETRIES:

                    wait_time = INITIAL_BACKOFF * (2 ** attempt)

                    print(
                        f"Rate limit encountered. "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                    continue

                return (
                    "LLM request failed because the API "
                    "rate limit was exceeded after multiple retries."
                )

            # --------------------------------------------
            # OTHER LLM FAILURE / TIMEOUT
            # --------------------------------------------

            if attempt < MAX_RETRIES:

                wait_time = INITIAL_BACKOFF * (2 ** attempt)

                print(
                    f"LLM request failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # --------------------------------------------
            # FINAL FAILURE
            # --------------------------------------------

            return (
                "LLM request failed after multiple attempts. "
                f"Error: {str(e)}"
            )

    return "LLM request failed unexpectedly."


# ============================================================
# RESEARCH NODE
# ============================================================

def research_node(state: AgentState):

    query = state["query"]

    # --------------------------------------------
    # SEARCH
    # --------------------------------------------

    try:

        search_results = get_search_tool().invoke(query)

    except Exception as e:

        search_results = (
            f"Search failed. "
            f"Proceed using the available information. "
            f"Error: {str(e)}"
        )

    # --------------------------------------------
    # LLM RESEARCH
    # --------------------------------------------

    prompt = RESEARCH_PROMPT.format(
        input=f"""
Query: {query}

Search Results:
{search_results}
"""
    )

    research_output = safe_llm_invoke(prompt)

    return {
        "research_output": research_output
    }


# ============================================================
# ANALYST NODE
# ============================================================

def analyst_node(state: AgentState):

    prompt = ANALYST_PROMPT.format(
        research_output=state["research_output"]
    )

    analysis_output = safe_llm_invoke(prompt)

    return {
        "analysis_output": analysis_output
    }


# ============================================================
# DECISION NODE
# ============================================================

def decision_node(state: AgentState):

    prompt = DECISION_PROMPT.format(
        analysis_output=state["analysis_output"]
    )

    decision_output = safe_llm_invoke(prompt)

    return {
        "decision_output": decision_output
    }


# ============================================================
# CRITIC NODE
# ============================================================

def critic_node(state: AgentState):

    prompt = CRITIC_PROMPT.format(
        decision_output=state["decision_output"]
    )

    critic_output = safe_llm_invoke(prompt)

    iterations = state.get("iterations", 0) + 1

    return {
        "critic_output": critic_output,
        "iterations": iterations
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

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


# ============================================================
# MAIN FUNCTION
# ============================================================

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
