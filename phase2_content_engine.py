"""
Phase 2: Autonomous Content Engine (LangGraph)
===============================================
A 3-node LangGraph state machine that:
  1. Decides what topic to post about (Decide Search)
  2. Fetches mock real-world context  (Web Search)
  3. Drafts a structured JSON post    (Draft Post)
"""

import json
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq          # pip install langchain-groq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ── LLM (swap to ChatOpenAI if you prefer) ─────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Search Tool
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web-search by returning hardcoded recent headlines
    based on keywords found in the query.
    """
    q = query.lower()

    if any(k in q for k in ["crypto", "bitcoin", "ethereum", "blockchain"]):
        return (
            "Bitcoin hits new all-time high amid regulatory ETF approvals. "
            "Ethereum layer-2 solutions cut gas fees by 90%."
        )
    if any(k in q for k in ["ai", "openai", "llm", "gpt", "model", "developer"]):
        return (
            "OpenAI releases GPT-5 with human-level reasoning benchmarks. "
            "AI coding assistants now write 40% of production code at major tech firms."
        )
    if any(k in q for k in ["stock", "market", "fed", "interest rate", "finance", "trading"]):
        return (
            "Fed signals two rate cuts in 2025 amid cooling inflation data. "
            "S&P 500 reaches record high on strong tech earnings."
        )
    if any(k in q for k in ["elon", "tesla", "spacex", "space", "musk"]):
        return (
            "SpaceX Starship completes first fully successful orbital mission. "
            "Tesla FSD achieves Level 4 autonomy in urban environments."
        )
    if any(k in q for k in ["climate", "environment", "nature", "pollution"]):
        return (
            "UN report: 2024 was the hottest year on record. "
            "Major corporations miss net-zero pledges for third consecutive year."
        )
    if any(k in q for k in ["regulation", "government", "policy", "law"]):
        return (
            "EU AI Act enforcement begins — big tech faces billion-dollar compliance costs. "
            "US Congress debates landmark social media privacy bill."
        )

    # fallback
    return (
        "Tech sector leads global market rally. "
        "AI adoption accelerates across all major industries in Q2 2025."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════════════

class BotState(TypedDict):
    bot_id: str
    persona: str
    search_query: Optional[str]
    search_results: Optional[str]
    topic: Optional[str]
    post_content: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Structured Output Schema  (enforces JSON output from the LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class PostOutput(BaseModel):
    bot_id: str = Field(description="The ID of the bot authoring the post")
    topic: str = Field(description="The topic of the post in 3-5 words")
    post_content: str = Field(description="The post text, max 280 characters, opinionated")


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 – Decide Search
# ═══════════════════════════════════════════════════════════════════════════════

def decide_search(state: BotState) -> BotState:
    """
    The LLM reads the bot's persona and decides what topic to post about today,
    then generates an appropriate search query.
    """
    print(f"\n[Node 1 - Decide Search]  bot={state['bot_id']}")

    prompt = (
        f"You are a social media bot with the following unwavering persona:\n"
        f"{state['persona']}\n\n"
        f"Decide ONE topic you want to post about today that fits your worldview.\n"
        f"Respond ONLY with valid JSON — no markdown, no explanation:\n"
        f'{{"topic": "<topic in 3-5 words>", "search_query": "<web search query>"}}'
    )

    response = llm.invoke(prompt)

    # Strip markdown fences if the model adds them
    raw = response.content.strip().strip("```json").strip("```").strip()
    data = json.loads(raw)

    print(f"  topic        : {data['topic']}")
    print(f"  search_query : {data['search_query']}")

    return {**state, "topic": data["topic"], "search_query": data["search_query"]}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 – Web Search
# ═══════════════════════════════════════════════════════════════════════════════

def web_search(state: BotState) -> BotState:
    """Runs the mock search tool with the query generated in Node 1."""
    print(f"\n[Node 2 - Web Search]")

    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  results : {results}")

    return {**state, "search_results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 – Draft Post
# ═══════════════════════════════════════════════════════════════════════════════

def draft_post(state: BotState) -> BotState:
    """
    Uses structured output (Pydantic + function calling) to guarantee the LLM
    returns a strict JSON object: {"bot_id", "topic", "post_content"}.
    """
    print(f"\n[Node 3 - Draft Post]")

    structured_llm = llm.with_structured_output(PostOutput)

    prompt = (
        f"You are a social media bot. Persona (NEVER break character):\n"
        f"{state['persona']}\n\n"
        f"Topic you chose: {state['topic']}\n"
        f"Breaking news context: {state['search_results']}\n\n"
        f"Write a highly opinionated, provocative post under 280 characters "
        f"that reflects your persona. bot_id is '{state['bot_id']}'."
    )

    result: PostOutput = structured_llm.invoke(prompt)

    print(f"  bot_id       : {result.bot_id}")
    print(f"  topic        : {result.topic}")
    print(f"  post_content : {result.post_content}")
    print(f"  char count   : {len(result.post_content)}")

    return {**state, "post_content": result.post_content}


# ═══════════════════════════════════════════════════════════════════════════════
# Build & Compile the LangGraph
# ═══════════════════════════════════════════════════════════════════════════════

def build_content_graph():
    """
    Graph topology:
        decide_search → web_search → draft_post → END
    """
    graph = StateGraph(BotState)

    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


# ── Quick demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS   # reuse persona definitions

    app = build_content_graph()

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"Running content engine for {bot_id}")
        print("="*60)

        initial_state: BotState = {
            "bot_id": bot_id,
            "persona": persona,
            "search_query": None,
            "search_results": None,
            "topic": None,
            "post_content": None,
        }

        final_state = app.invoke(initial_state)

        print(f"\n📤 Final JSON output:")
        print(json.dumps({
            "bot_id": final_state["bot_id"],
            "topic": final_state["topic"],
            "post_content": final_state["post_content"],
        }, indent=2))
