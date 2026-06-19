import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --------------------------
# Load PDFs
# --------------------------
loader = PyPDFDirectoryLoader("./documents")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# --------------------------
# Split into chunks
# --------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# --------------------------
# Embedding model
# --------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# --------------------------
# Chroma vector store
# --------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="pdf_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --------------------------
# LLM
# --------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# --------------------------
# Prompt
# --------------------------
template = """
Answer the question only from the provided context.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# --------------------------
# Chat loop
# --------------------------
while True:
    query = input("\nAsk a question (q to quit): ")

    if query.lower() == "q":
        break

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    print("\nAnswer:\n")
    print(response.content)