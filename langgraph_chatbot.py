import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# load and set keys
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

# Initialize llm
llm = ChatGroq(groq_api_key=groq_api_key,model_name= "Gemma2-9b-It")

# Define State
class State(TypedDict):
    messages:Annotated[list,add_messages]

# initilize Graph Builder
graph_builder = StateGraph(State)

# Define  chatbot with state
def Chatbot(state:State):
    return {"messages":llm.invoke(state['messages'])}

graph_builder.add_node("chatbot",Chatbot)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

# user input
while True:
    user_input = input("User : ")
    if user_input.lower() in ["quite","q"]:
        print("Good bye")
        break
    for event in graph.stream({"messages":{"user",user_input}}):
        print(event.values())
        for value in event.values():
            print(value['messages'])
            print("Assistant:",value["messages"].content)


#Note : Use below code on colab for dislay node flow
"""
  from IPython.display import Image,display
try:
  display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
  print(e)
"""

