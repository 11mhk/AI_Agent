from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from datetime import datetime

# Save tool
@tool
def save_tool(data: str, filename: str = "research_output.txt") -> str:
    """Saves structured research data to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

# Search tool
_search_wrapper = DuckDuckGoSearchAPIWrapper()

@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return _search_wrapper.run(query)

# Wikipedia tool
_wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia for information on a topic."""
    return _wiki_wrapper.run(query)
