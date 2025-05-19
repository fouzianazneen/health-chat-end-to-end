
# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from src.prompt import prompt_template
# from pinecone import Pinecone
# # Install the new package if needed: pip install langchain-pinecone
# from langchain_pinecone import PineconeVectorStore

# import os

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# # Pinecone setup
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
# index_name = "medical-chatbot"

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Get the specific index
# pinecone_index = pc.Index(index_name)

# # Download embeddings
# embeddings = download_hugging_face_embeddings()

# # Initialize vectorstore with the updated PineconeVectorStore class
# docsearch = PineconeVectorStore(
#     index=pinecone_index,
#     embedding=embeddings,
#     text_key="text",
#     namespace="default"
# )

# # Setup prompt
# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )

# # LLM config using CTransformers
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
#     chain_type_kwargs={"prompt": PROMPT}
# )

# # Routes
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
#     result = qa({"query": input})
#     return str(result["result"])

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)













from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
import os

# Import pinecone
import pinecone
print(f"Pinecone version: {pinecone.__version__}")

# For production, consider using an API-based model
from langchain_community.llms import CTransformers
# Or use OpenAI API
# from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Pinecone setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
index_name = "medical-chatbot"

# Initialize Pinecone client - try the correct approach for version 6.0.2
try:
    # For version 6.0.2
    from pinecone import Pinecone as PineconeSDK
    pc = PineconeSDK(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name)
    print("Connected to Pinecone using Pinecone SDK class")
except (ImportError, AttributeError) as e:
    print(f"Error with Pinecone SDK import: {e}")
    try:
        # Alternative approach
        pc = pinecone.GRPCIndex(index_name)
        pinecone_index = pc
        print("Connected to Pinecone using GRPCIndex")
    except (ImportError, AttributeError) as e:
        print(f"Error with GRPCIndex: {e}")
        try:
            # Very old approach
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
            pinecone_index = pinecone.Index(index_name)
            print("Connected to Pinecone using old init method")
        except Exception as e:
            print(f"Error with old init method: {e}")
            print("Please install a compatible version of Pinecone: pip install 'pinecone-client>=2.2.1,<3.0.0'")
            print("Current Pinecone methods available:", dir(pinecone))
            raise

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize vectorstore with the updated PineconeVectorStore class
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    text_key="text",
    namespace="default"
)

# Setup prompt
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# LLM config using CTransformers (consider switching to an API-based model for deployment)
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # This will be challenging on Render
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
    chain_type_kwargs={"prompt": PROMPT}
)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = qa.invoke({"query": user_input})  # Fixed deprecated method
    return jsonify({"response": result["result"]})

@app.route("/get", methods=["GET", "POST"])
def get_response():
    msg = request.form["msg"]
    input = msg
    result = qa.invoke({"query": input})  # Fixed deprecated method
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)