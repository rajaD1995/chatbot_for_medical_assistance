from dotenv import load_dotenv
import os
from src.helper import pdf_loader,filter_doc,text_split,download_embedding_model
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # to build vector DB
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # to build LLM
# Embedding model install locally so no need of API key

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = pdf_loader(r"data")
minimul_doc = filter_doc(extracted_data)
text_chunks = text_split(minimul_doc)

model = download_embedding_model()

pinecone_api_key = PINECONE_API_KEY
# authenticate pinecone account
pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chatbot"

# Creating index in pinecone
if not pc.has_index(index_name):  # Check is medical-chatbot is not present in pinecone
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension of embedding model which is BAAI/bge-small-en-v1.5
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )


index = pc.Index(index_name)

# save the embedding in the index of pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=model,
    index_name= index_name
)
