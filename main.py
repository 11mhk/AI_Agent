from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

from google import genai
from ResearchTools import search_tool, wiki_tool, save_tool
from MathsTools import addition

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

class MathsResponse(BaseModel):
    result: float
    tools_used: list[str]
    
class ManagerResponse(BaseModel):
    decision: int

client = genai.Client()
#client = genai.Client()

llm = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it")
#llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


ManagerParser = PydanticOutputParser(pydantic_object=ManagerResponse)
MathsParser= PydanticOutputParser(pydantic_object=MathsResponse)
ResearchParser = PydanticOutputParser(pydantic_object=ResearchResponse)

#manager
system_prompt_Manager = """
You are a manager who will decide the action based on the user input.Give decision as 1 if the query is research related and give the output as 2 if the query is maths related.
Wrap the output in this format and provide no other text\n{format_instructions}
""".format(format_instructions=ManagerParser.get_format_instructions())

#reserach
system_prompt_research = """
You are a Research assistant that will help the user do research.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text\n{format_instructions}
""".format(format_instructions=ResearchParser.get_format_instructions())

#maths
system_prompt_maths = """
You are a mathematic assistant that will help in mathematical calculations.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text\n{format_instructions}
""".format(format_instructions=MathsParser.get_format_instructions())


tools_research = [search_tool, wiki_tool, save_tool]
tools_maths = [ addition]

agent_manager = create_agent(
    model=llm,
    system_prompt=system_prompt_Manager

)


query = input("How can i help you? (research and maths) ")
raw_response = agent_manager.invoke({
    "messages": [HumanMessage(content=query)]
})

#print(raw_response)


try:
    output = raw_response["messages"][-1].content

    # Anthropic may return a list of content blocks
    if isinstance(output, list):
        output = next((block["text"] for block in output if block.get("type") == "text"), "")

    structured_response = ManagerParser.parse(output)
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)




# condiions 
if  structured_response.decision == 1:
    #call research model
    print("calling researcher")
    agent_research = create_agent(
    model=llm,
    tools=tools_research,
    system_prompt=system_prompt_research
)
    raw_response = agent_research.invoke({
    "messages": [HumanMessage(content=query)]})
    try:
        output = raw_response["messages"][-1].content
        # Anthropic may return a list of content blocks
        if isinstance(output, list):
            output = next((block["text"] for block in output if block.get("type") == "text"), "")

            structured_response = ResearchParser.parse(output)
            print(structured_response)

    except Exception as e:
        print("Error parsing response:", e)
        print("Raw Response:", raw_response)




elif structured_response.decision == 2:
    print("Calling maths")

    agent_maths = create_agent(
        model=llm,
        tools=tools_maths,
        system_prompt=system_prompt_maths
    )

    raw_response = agent_maths.invoke({
        "messages": [HumanMessage(content=query)]
    })

    try:
        output = raw_response["messages"][-1].content

        if isinstance(output, list):
            output = next(
                (block["text"] for block in output if block.get("type") == "text"),
                ""
            )

        structured_response = MathsParser.parse(output)
        print(structured_response)

    except Exception as e:
        print("Error parsing response:", e)
        print("Raw Response:", raw_response)


else:
    print("cannot process your request")

