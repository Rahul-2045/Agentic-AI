from pathlib import Path
from langchain_core.tools import tool

KB_DIR = Path("knowledge_base")

@tool
def sop_search_tool(query: str) -> str:
    """Search trip monitoring SOP or knowledge base text for guidance."""
    texts = []
    for file in KB_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        if any(word.lower() in content.lower() for word in query.split()):
            texts.append(f"Source: {file.name}\n{content[:2000]}")
    if not texts:
        return "No matching SOP content found."
    return "\n\n".join(texts)

def get_rag_tools():
    return [sop_search_tool]
