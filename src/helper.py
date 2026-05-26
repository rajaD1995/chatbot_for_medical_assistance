from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

# PDF loader function
def pdf_loader(file_path):
    loader = DirectoryLoader(
        path=file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    document = loader.load()
    return document

# metadata: You can see the metadata. The important one is the page_content and source
# Cleaning
def filter_doc(docs: List[Document])->List[Document]:
    '''
    Given a list document object, retun a list of document object
    containing only source and original page content as meta data
    '''
    minimul_doc: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimul_doc.append(
            Document(
                page_content=doc.page_content,
                metadata={'source':src}
            )
        )
    return minimul_doc


# Chunking
def text_split(minimul_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 50,
    )
    texts_chunks = text_splitter.split_documents(minimul_doc)
    return texts_chunks


# Embedding
def download_embedding_model():
    '''
    Download and return the BAAI bge-small-en embedding model
    '''
    # Make sure to use the exact official model name
    model_name = "BAAI/bge-small-en-v1.5" # dimension = 384, token = 512
    
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True} # Recommended for similarity search
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings
