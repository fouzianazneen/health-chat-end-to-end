
# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
<<<<<<< HEAD
=======
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from pinecone import Pinecone as PineconeClient
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
<<<<<<< HEAD
# from src.prompt import prompt_template
# from pinecone import Pinecone
# # Install the new package if needed: pip install langchain-pinecone
# from langchain_pinecone import PineconeVectorStore

=======
# from src.prompt import *
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
# import os

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

<<<<<<< HEAD
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
=======
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
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )
<<<<<<< HEAD

# # LLM config using CTransformers
=======
# chain_type_kwargs = {"prompt": PROMPT}

# # LLM configuration
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
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
<<<<<<< HEAD
#     chain_type_kwargs={"prompt": PROMPT}
# )

# # Routes
=======
#     chain_type_kwargs=chain_type_kwargs
# )






>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
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
<<<<<<< HEAD
#     result = qa({"query": input})
#     return str(result["result"])

=======
#     print("User input:", input)
#     result = qa({"query": input})
#     print("Response:", result["result"])
#     return str(result["result"])


>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)











<<<<<<< HEAD


from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_core.prompts import PromptTemplate
=======
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
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

<<<<<<< HEAD
# Initialize vectorstore with the updated PineconeVectorStore class
from langchain_pinecone import PineconeVectorStore
=======
# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot"
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49

docsearch = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    text_key="text",
    namespace="default"
)
<<<<<<< HEAD

# Setup prompt
=======
 
# Prompt setup
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

<<<<<<< HEAD
# LLM config using CTransformers (consider switching to an API-based model for deployment)
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # This will be challenging on Render
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8
    }
)
=======
# LLM configuration
try:
    import os.path
    model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
    
    if os.path.isfile(model_path):
        llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                'max_new_tokens': 512,
                'temperature': 0.8
            }
        )
    else:
        from langchain_community.llms import HuggingFaceHub
        print(f"Warning: Model file {model_path} not found. Falling back to HuggingFaceHub.")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_KEY", "")
        llm = HuggingFaceHub(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            model_kwargs={"temperature": 0.8, "max_new_tokens": 512}
        )
except Exception as e:
    from langchain.llms import OpenAI
    print(f"Warning: Failed to load LLM: {str(e)}. Falling back to OpenAI.")
    llm = OpenAI(temperature=0.8)
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

<<<<<<< HEAD
# Routes
=======
>>>>>>> 41c3c380f9fbe7c2a97004fa79baf3eccb4afe49
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