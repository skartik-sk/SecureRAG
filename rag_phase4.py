import os
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ Error: GROQ_API_KEY is missing!")

converter = DocumentConverter()
result = converter.convert("https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md")
markdown_content = result.document.export_to_markdown()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown_content)

for doc in docs:
    doc.metadata["role"] = "user"
    doc.metadata["category"] = "ecomm_docs"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
connection_string = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="ecommerce_docs",
    connection=connection_string,
    use_jsonb=True,
)

vectorstore.add_documents(docs)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question accurately using ONLY the context provided below.
If you do not know the answer based on the context, say "I don't know."

Context: {context}
Question: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_role: str

def secure_rag_query_node(state: AgentState):
    user_query = state["messages"][-1].content
    current_user_role = state["user_role"]
    
    validation_results = vectorstore.similarity_search_with_score(user_query, k=1)
    if not validation_results:
        return {"messages": [AIMessage(content="Access Blocked: I can only answer documentation questions.")]}
        
    closest_doc, score = validation_results[0]
    if score > 0.95:  
        return {"messages": [AIMessage(content="Access Blocked: I am strictly programmed to answer documentation questions only.")]}

    secured_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"role": current_user_role}
        }
    )
    
    secured_chain = create_retrieval_chain(secured_retriever, combine_docs_chain)
    execution_response = secured_chain.invoke({"input": user_query})
    
    return {"messages": [AIMessage(content=execution_response['answer'])]}

workflow = StateGraph(AgentState)
workflow.add_node("rag_agent_node", secure_rag_query_node)

workflow.add_edge(START, "rag_agent_node")
workflow.add_edge("rag_agent_node", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session_user_kartik_101"}}

print("\n🚀 Turn 1: Asking the initial question...")
payload_1: AgentState = {
    "messages": [HumanMessage(content="What is this repository about?")],
    "user_role": "user"
}
output_1 = app.invoke(payload_1, config=config)
print(f"Agent Response: {output_1['messages'][-1].content}")

print("\n💬 Turn 2: Asking a follow-up question (Testing Context Memory)...")
payload_2: AgentState = {
    "messages": [HumanMessage(content="Does it include any agent blueprints or implementations?")],
    "user_role": "user"
}
output_2 = app.invoke(payload_2, config=config)
print(f"Agent Response: {output_2['messages'][-1].content}")