from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
import os


load_dotenv()

# load and set keys
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY") 


# Initialize tools wrapper and query
arxix_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxix_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

response = wiki_tool.invoke("Who is salman Khan")
print("Response ==> ",response)

# We Can Initilize multiple tools
tools = [wiki_tool]


# Initialize State
class State(TypedDict):
    messages: Annotated[list,add_messages]

graph_builder = StateGraph(State)


#Initialize llm 
llm = ChatGroq(groq_api_key=groq_api_key,model="Gemma2-9b-It")

#Bind llm with tools
llm_with_tools = llm.bind_tools(tools=tools)

# define chatbot
def chatbot(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Define node and ages 
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
tool_node = ToolNode(tools=tools)

graph_builder.add_node("tools",tool_node)
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()


# Note : for graphical represent use below code in colab

"""
  # Display Graph
from IPython.display import Image,display

try:
  display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
  pass
"""

user_input = "Hi ! ,I am John Cena"

events = graph.stream({"messages":[("user",user_input)]},stream_mode="values")

for event in events:
   print(event["messages"])






