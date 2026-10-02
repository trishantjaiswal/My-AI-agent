import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables from .env file
load_dotenv()

# Define the output structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the free Gemini model (Gemini 3.6 Flash is fast and free-tier friendly)
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

# Set up the Pydantic parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"), #coming from user
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent = agent , tools = tools , verbose = True)
query = input("What can I help you with?")
raw_response = agent_executor.invoke({"query": query})

# Extract the text string from the 'output' key and parse it into your Pydantic model
# 1. Get the raw output block
output_data = raw_response["output"]

# 2. If Gemini returned a list of content blocks, extract the text from the first one
if isinstance(output_data, list):
    output_data = output_data[0]["text"]

# 3. Parse the clean text string
structured_response = parser.parse(output_data)

# Now you can print specific fields!
print("Topic:", structured_response.topic)
print("Summary:", structured_response.summary)
print("Sources:", structured_response.sources)