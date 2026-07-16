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


load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ Error: GROQ_API_KEY is missing!")

print("env loaded")

converter = DocumentConverter()
result = converter.convert("https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md")
markdown_content = result.document.export_to_markdown()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown_content)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

connection_string = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="ecommerce_docs",
    connection=connection_string,
    use_jsonb=True,
)

vectorstore.add_documents(docs)
print("added to vector pg")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question accurately using ONLY the context provided below.
If you do not know the answer based on the context, say "I don't know."

Context: {context}
Question: {input}
""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query = "What is this repository about?"
response = retrieval_chain.invoke({"input": query})

print(f"\nQuestion: {query}")
print(f"Answer: {response['answer']}")