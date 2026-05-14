from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from datetime import datetime

#addition tool
@tool
def addition(a: int, b: int)-> int:
    """Use this tool for addition of numbers."""
    return a+b

