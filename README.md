# langGraph-GenAI
# **LangGraph Chatbot with Tools and Vector Store Integration**

This repository contains the implementation of an advanced chatbot using **LangGraph**, **LangChain**, and **ChatGroq**, designed to support multi-turn conversations, integrate with external APIs like Wikipedia and Arxiv, and retrieve domain-specific knowledge from AstraDB VectorStore.

---

## **Features**
- **Stateful Chatbot**: Handles multi-turn conversations using LangGraph's stateful workflow.
- **Tool Integration**: Incorporates tools like Wikipedia and Arxiv APIs for real-time external data retrieval.
- **Retrieval-Augmented Generation (RAG)**: Leverages AstraDB VectorStore for precise domain-specific answers.
- **Dynamic Question Routing**: Automatically routes user queries to the most relevant source (vector store or external API).
- **Graph Visualization**: Generates graphical representations of the chatbot workflow for easy debugging and understanding.

---

## **Installation**
python install -r requirements.txt

### **Prerequisites**
- Python 3.9 or higher
- Clone the repository:
  ```bash
  git clone <repository_url>
  cd <repository_name>
