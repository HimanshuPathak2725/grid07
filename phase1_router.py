"""
Phase 1: Vector-Based Persona Matching (The Router)
=====================================================
Embeds bot personas into ChromaDB and routes incoming posts
to only the bots that "care" about the topic via cosine similarity.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# ── Embedding model (runs locally, no API key needed) ──────────────────────────
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Bot Personas ────────────────────────────────────────────────────────────────
BOT_PERSONAS: Dict[str, str] = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}

# ── Build in-memory ChromaDB with cosine similarity ────────────────────────────
def build_persona_store() -> chromadb.Collection:
    """Creates an in-memory ChromaDB collection and stores bot persona embeddings."""
    client = chromadb.Client()  # ephemeral / in-memory

    # cosine space: distance = 1 - cosine_similarity
    collection = client.get_or_create_collection(
        name="bot_personas",
        metadata={"hnsw:space": "cosine"},
    )

    for bot_id, persona_text in BOT_PERSONAS.items():
        embedding = EMBED_MODEL.encode(persona_text).tolist()
        collection.add(
            ids=[bot_id],
            embeddings=[embedding],
            documents=[persona_text],
            metadatas=[{"bot_id": bot_id}],
        )

    print(f"[Phase 1] Stored {collection.count()} bot personas in vector store.\n")
    return collection


# ── Router function ─────────────────────────────────────────────────────────────
def route_post_to_bots(
    post_content: str,
    collection: chromadb.Collection,
    threshold: float = 0.20,   # NOTE: all-MiniLM-L6-v2 rarely exceeds 0.6 for
                                # semantically related but not identical text.
                                # Start at 0.20 and raise to taste.
) -> List[Dict]:
    """
    Embeds a post and returns bots whose persona cosine-similarity exceeds threshold.

    Args:
        post_content: The incoming social-media post text.
        collection:   The ChromaDB collection holding persona embeddings.
        threshold:    Minimum cosine similarity (0–1) to consider a bot "interested".

    Returns:
        List of dicts: [{"bot_id": str, "similarity": float, "persona": str}, ...]
    """
    post_embedding = EMBED_MODEL.encode(post_content).tolist()

    results = collection.query(
        query_embeddings=[post_embedding],
        n_results=len(BOT_PERSONAS),          # check all bots
        include=["distances", "documents", "metadatas"],
    )

    matched: List[Dict] = []
    for bot_id, distance, document in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        similarity = 1.0 - distance           # cosine distance → similarity
        status = "✅ MATCHED" if similarity >= threshold else "❌ skipped"
        print(
            f"  {status}  {bot_id}  similarity={similarity:.4f}  "
            f"(threshold={threshold})"
        )
        if similarity >= threshold:
            matched.append(
                {"bot_id": bot_id, "similarity": round(similarity, 4), "persona": document}
            )

    return matched


# ── Quick demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = build_persona_store()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high — regulators are panicking.",
        "The Fed just raised interest rates by 50 basis points.",
        "Nature documentaries prove we need to protect rainforests immediately.",
    ]

    for post in test_posts:
        print(f"\n📨 Post: \"{post}\"")
        matches = route_post_to_bots(post, store, threshold=0.30)
        if matches:
            print(f"  → Routed to: {[m['bot_id'] for m in matches]}")
        else:
            print("  → No bots matched this post.")
