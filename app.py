
# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from pinecone import Pinecone as PineconeClient
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from src.prompt import *
# import os

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # e.g., aws-us-west-2

# # Split the env into cloud and region
# CLOUD, REGION = PINECONE_API_ENV.split('-')[0], '-'.join(PINECONE_API_ENV.split('-')[1:])

# # Initialize embedding model
# embeddings = download_hugging_face_embeddings()

# # Initialize Pinecone client
# pc = PineconeClient(api_key=PINECONE_API_KEY)
# index_name = "chatbot"

# docsearch = LangchainPinecone.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
#     namespace="default"  # optional, use only if you are using namespaces
# )
 


# # Prompt setup
# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

# # LLM configuration
# llm = CTransformers(
#     model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#     model_type="llama",
#     config={
#         'max_new_tokens': 512,
#         'temperature': 0.8
#     }
# )

# # QA Chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs
# )






# @app.route("/")
# def index():
#     return render_template("chat.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.json.get("message")
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     result = qa({"query": user_input})
#     return jsonify({"response": result["result"]})

# @app.route("/get", methods=["GET", "POST"])
# def get_response():
#     msg = request.form["msg"]
#     input = msg
#     print("User input:", input)
#     result = qa({"query": input})
#     print("Response:", result["result"])
#     return str(result["result"])


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)











from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # e.g., aws-us-west-2

# Split the env into cloud and region
CLOUD, REGION = PINECONE_API_ENV.split('-')[0], '-'.join(PINECONE_API_ENV.split('-')[1:])

# Initialize embedding model
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "chatbot"

docsearch = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="default"  # optional, use only if you are using namespaces
)
 
# Prompt setup
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# LLM configuration
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8
    }
)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = qa({"query": user_input})
    return jsonify({"response": result["result"]})

@app.route("/get", methods=["GET", "POST"])
def get_response():
    msg = request.form["msg"]
    input = msg
    print("User input:", input)
    result = qa({"query": input})
    print("Response:", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
