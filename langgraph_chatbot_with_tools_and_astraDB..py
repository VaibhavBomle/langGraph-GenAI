import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import os
from langchain_groq import ChatGroq
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from pprint import pprint


groq_api_key = os.getenv("GROQ_API_KEY")


# Connection of ASTRA DB
ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)




# Create Index

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

## load
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]
print(doc_list)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
docs_split = text_splitter.split_documents(doc_list)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector store
astra_vector_store = Cassandra(embedding=embeddings,
                               table_name = "qa_mini_demo",
                               session=None,
                               keyspace=None)


astra_vector_store.add_documents(docs_split)
print("Inserted %i headlines. "% len(docs_split))
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

retriver = astra_vector_store.as_retriever()
retriver.invoke("What is Generative AI")



## Langgraph

# Data model
class RouteQuery(BaseModel):
  """Route a user query to the most relevant datasource."""

  datasource: Literal["vectorstore","wiki_search"] = Field(
      ...,
      description="Given a user question choose to route it to wikipedia or a vectorstore"
  )


llm = ChatGroq(groq_api_key=groq_api_key,model="Llama-3.1-70b-versatile")


structured_llm_route = llm.with_structured_output(RouteQuery)


# Defining a Prompt

system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents,prompt engineering , and adversarial attacks.
Use the vectorstore for questios on these topics.Otherwise, use-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","{question}")
    ]
)
question_router = route_prompt | structured_llm_route


# Check Source
print(question_router.invoke(
    {
        "question":"What is Agent?"
    }
))


# Check Source
print(question_router.invoke(
    {
        "question":"Who is PM of India?"
    }
))


api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki.run("Tell me about Salman Khan")



# LangGraph  GraphState
class GraphState(TypedDict):
  """
  Represents the state of our group

  Attributes:
      question: question
      generation: LLM Generation
      documents: list of documents
  """
  question : str
  generation: str
  documents: List[str]


def retriever(state):
  """
  Retrieve documents

  Args:
      state (dict): The current graph state

  Returns:
      state (dict): New key added to state, documents, that contains retrieved documents
  """
  print("---RETRIEVER---")
  question = state["question"]

  # Retriever
  documents = retriever.invoke(question)
  return {"documents": documents, "question" : question}




def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
       state (dict): The current graph state

    Returns:
       state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    question = state["question"]
    print(question)

    #Wiki search
    docs = wiki.invoke({"query": question})
    wiki_results = docs
    wiki_results =Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}





# Edges

def route_question(state):
  """
  Route question to wiki search or RAG.

  Args:
      state (dict): The current graph state

  Returns:
      str: Next node to call
  """

  print("---ROUTE QUESTION---")
  question = state["question"]
  source = question_router.invoke({"question": question})
  datasource = source.datasource if hasattr(source, "datasource") else ""  

  if datasource == "wiki_search":
    print("---ROUTE QUESTION TO WIKI SEARCH")
    return "wiki_search"
  elif datasource == "vectorstore":
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"
  

# Initialize graph flow

workflow = StateGraph(GraphState)

workflow.add_node("wiki_search",wiki_search) # web search
workflow.add_node("retriver",retriver) # retriever

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retriver"
    }
)

workflow.add_edge("retriver",END)
workflow.add_edge("wiki_search",END)

# Compile
app = workflow.compile()



"""
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

"""


# Run
inputs = {
    "question": "What is agent?"
}

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        
    pprint("\n---\n")

# Final generation
pprint(value['documents'][0].dict()['metadata']['description'])




