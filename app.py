from flask import Flask,render_template,jsonify,request
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from langchain.chains import create_retrieval_chain

import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # to build vector DB
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # to build LLM

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding = download_embedding_model()
index_name = "medical-chatbot"

# Load existing embedding from existing index of pinecone
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# So we get the embedding from the pinecone

# Chain
retriver = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

chatmodel = ChatOpenAI(model="gpt-4o",temperature=0)



# Chatbot take two prompts one system prompt and user prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatmodel,prompt)
rag_chain = create_retrieval_chain(retriver,question_answer_chain)

# Basic route, default route
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/chat",methods=["POST"])
def chat():
    # Change this line - get JSON data instead of form data
    data = request.get_json()  # Get JSON from request body
    msg = data.get("message")   # HTML sends "message", not "msg"
    
    print(f"User input: {msg}")
    response = rag_chain.invoke({"input":msg})
    print("Response: ",response['answer'])

    # Return JSON, not string
    return jsonify({"response": response['answer']})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)
