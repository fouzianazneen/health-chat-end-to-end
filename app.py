



# # # # # from flask import Flask, render_template, jsonify, request
# # # # # from src.helper import download_hugging_face_embeddings
# # # # # from langchain_core.prompts import PromptTemplate
# # # # # from langchain.chains import RetrievalQA
# # # # # from dotenv import load_dotenv
# # # # # from src.prompt import prompt_template
# # # # # import os

# # # # # # Import pinecone
# # # # # import pinecone
# # # # # print(f"Pinecone version: {pinecone.__version__}")

# # # # # # For production, consider using an API-based model
# # # # # from langchain_community.llms import CTransformers
# # # # # # Or use OpenAI API
# # # # # # from langchain_openai import ChatOpenAI

# # # # # # Load environment variables
# # # # # load_dotenv()

# # # # # app = Flask(__name__)

# # # # # # Pinecone setup
# # # # # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# # # # # PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
# # # # # index_name = "medical-chatbot"

# # # # # # Initialize Pinecone client - try the correct approach for version 6.0.2
# # # # # try:
# # # # #     # For version 6.0.2
# # # # #     from pinecone import Pinecone as PineconeSDK
# # # # #     pc = PineconeSDK(api_key=PINECONE_API_KEY)
# # # # #     pinecone_index = pc.Index(index_name)
# # # # #     print("Connected to Pinecone using Pinecone SDK class")
# # # # # except (ImportError, AttributeError) as e:
# # # # #     print(f"Error with Pinecone SDK import: {e}")
# # # # #     try:
# # # # #         # Alternative approach
# # # # #         pc = pinecone.GRPCIndex(index_name)
# # # # #         pinecone_index = pc
# # # # #         print("Connected to Pinecone using GRPCIndex")
# # # # #     except (ImportError, AttributeError) as e:
# # # # #         print(f"Error with GRPCIndex: {e}")
# # # # #         try:
# # # # #             # Very old approach
# # # # #             pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
# # # # #             pinecone_index = pinecone.Index(index_name)
# # # # #             print("Connected to Pinecone using old init method")
# # # # #         except Exception as e:
# # # # #             print(f"Error with old init method: {e}")
# # # # #             print("Please install a compatible version of Pinecone: pip install 'pinecone-client>=2.2.1,<3.0.0'")
# # # # #             print("Current Pinecone methods available:", dir(pinecone))
# # # # #             raise

# # # # # # Download embeddings
# # # # # embeddings = download_hugging_face_embeddings()

# # # # # # Initialize vectorstore with the updated PineconeVectorStore class
# # # # # from langchain_pinecone import PineconeVectorStore

# # # # # docsearch = PineconeVectorStore(
# # # # #     index=pinecone_index,
# # # # #     embedding=embeddings,
# # # # #     text_key="text",
# # # # #     namespace="default"
# # # # # )

# # # # # # Setup prompt
# # # # # PROMPT = PromptTemplate(
# # # # #     template=prompt_template,
# # # # #     input_variables=["context", "question"]
# # # # # )

# # # # # # LLM config using CTransformers (consider switching to an API-based model for deployment)
# # # # # llm = CTransformers(
# # # # #     model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # This will be challenging on Render
# # # # #     model_type="llama",
# # # # #     config={
# # # # #         'max_new_tokens': 512,
# # # # #         'temperature': 0.8
# # # # #     }
# # # # # )

# # # # # # QA Chain
# # # # # qa = RetrievalQA.from_chain_type(
# # # # #     llm=llm,
# # # # #     chain_type="stuff",
# # # # #     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
# # # # #     return_source_documents=True,
# # # # #     chain_type_kwargs={"prompt": PROMPT}
# # # # # )

# # # # # # Routes
# # # # # @app.route("/")
# # # # # def index():
# # # # #     return render_template("chat.html")

# # # # # @app.route("/chat", methods=["POST"])
# # # # # def chat():
# # # # #     user_input = request.json.get("message")
# # # # #     if not user_input:
# # # # #         return jsonify({"error": "No message provided"}), 400

# # # # #     result = qa.invoke({"query": user_input})  # Fixed deprecated method
# # # # #     return jsonify({"response": result["result"]})

# # # # # @app.route("/get", methods=["GET", "POST"])
# # # # # def get_response():
# # # # #     msg = request.form["msg"]
# # # # #     input = msg
# # # # #     result = qa.invoke({"query": input})  # Fixed deprecated method
# # # # #     return str(result["result"])

# # # # # if __name__ == '__main__':
# # # # #     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)












# # from flask import Flask, render_template, jsonify, request
# # from dotenv import load_dotenv
# # from langchain.chains import RetrievalQA
# # from langchain_core.prompts import PromptTemplate
# # from langchain_community.llms import CTransformers
# # from langchain_pinecone import PineconeVectorStore
# # from pinecone import Pinecone, ServerlessSpec
# # from src.helper import download_hugging_face_embeddings
# # from src.prompt import prompt_template
# # import os

# # # Load environment variables
# # load_dotenv()

# # # Flask app
# # app = Flask(__name__)

# # # Pinecone setup
# # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# # index_name = "medical-chatbot"

# # # Initialize Pinecone client (v2.2.4)
# # pc = Pinecone(api_key=PINECONE_API_KEY)

# # # Connect to existing index
# # pinecone_index = pc.Index(index_name)

# # # Download HuggingFace embeddings
# # embeddings = download_hugging_face_embeddings()

# # # Setup vector store
# # docsearch = PineconeVectorStore(
# #     index=pinecone_index,
# #     embedding=embeddings,
# #     text_key="text",
# #     namespace="default"
# # )

# # # Prompt template
# # PROMPT = PromptTemplate(
# #     template=prompt_template,
# #     input_variables=["context", "question"]
# # )

# # # Load LLM with CTransformers (LLaMA 2)
# # llm = CTransformers(
# #     model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
# #     model_type="llama",
# #     config={
# #         'max_new_tokens': 512,
# #         'temperature': 0.8
# #     }
# # )

# # # Setup RetrievalQA chain
# # qa = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",
# #     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
# #     return_source_documents=True,
# #     chain_type_kwargs={"prompt": PROMPT}
# # )

# # # Flask routes
# # @app.route("/")
# # def index():
# #     return render_template("chat.html")

# # @app.route("/chat", methods=["POST"])
# # def chat():
# #     user_input = request.json.get("message")
# #     if not user_input:
# #         return jsonify({"error": "No message provided"}), 400

# #     result = qa.invoke({"query": user_input})
# #     return jsonify({"response": result["result"]})

# # @app.route("/get", methods=["GET", "POST"])
# # def get_response():
# #     msg = request.form["msg"]
# #     result = qa.invoke({"query": msg})
# #     return str(result["result"])

# # # Run app
# # if __name__ == '__main__':
# #     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
















# from flask import Flask, render_template, jsonify, request
# from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
# from src.helper import download_hugging_face_embeddings
# from src.prompt import prompt_template
# import os
# import requests
# from pathlib import Path

# # Load environment variables
# load_dotenv()

# # Flask app
# app = Flask(__name__)

# # Pinecone setup
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# index_name = "medical-chatbot"

# # Initialize Pinecone client (v2.2.4)
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Connect to existing index
# pinecone_index = pc.Index(index_name)

# # Download HuggingFace embeddings
# embeddings = download_hugging_face_embeddings()

# # Setup vector store
# docsearch = PineconeVectorStore(
#     index=pinecone_index,
#     embedding=embeddings,
#     text_key="text",
#     namespace="default"
# )

# # Prompt template
# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )

# def download_model():
#     """Download the model file if it doesn't exist"""
#     model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
    
#     if os.path.exists(model_path):
#         print(f"Model already exists at {model_path}")
#         return model_path
    
#     # Create model directory
#     os.makedirs("model", exist_ok=True)
    
#     # Model download URL (you need to replace this with actual URL)
#     # Option 1: Direct download URL if you have one
#     model_url = os.environ.get('MODEL_DOWNLOAD_URL')
    
#     if model_url:
#         print("Downloading model file...")
#         try:
#             response = requests.get(model_url, stream=True)
#             response.raise_for_status()
            
#             with open(model_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
            
#             print(f"Model downloaded successfully to {model_path}")
#             return model_path
#         except Exception as e:
#             print(f"Failed to download model: {e}")
#             return None
    
#     # Option 2: Use HuggingFace Hub to download
#     try:
#         from huggingface_hub import hf_hub_download
#         print("Attempting to download from HuggingFace Hub...")
        
#         model_path = hf_hub_download(
#             repo_id="TheBloke/Llama-2-7B-Chat-GGML",
#             filename="llama-2-7b-chat.ggmlv3.q4_0.bin",
#             local_dir="model",
#             local_dir_use_symlinks=False
#         )
#         print(f"Model downloaded from HuggingFace Hub to {model_path}")
#         return model_path
        
#     except Exception as e:
#         print(f"Failed to download from HuggingFace Hub: {e}")
#         return None

# def initialize_llm():
#     """Initialize the LLM with error handling"""
#     model_path = download_model()
    
#     if not model_path or not os.path.exists(model_path):
#         print("Model not available, using fallback response")
#         return None
    
#     try:
#         llm = CTransformers(
#             model=model_path,
#             model_type="llama",
#             config={
#                 'max_new_tokens': 512,
#                 'temperature': 0.8,
#                 'context_length': 2048,  # Reduce context to save memory
#                 'threads': 1,  # Use single thread to save memory
#             }
#         )
#         print("LLM initialized successfully")
#         return llm
#     except Exception as e:
#         print(f"Failed to initialize LLM: {e}")
#         return None

# # Initialize LLM
# llm = initialize_llm()

# # Setup RetrievalQA chain if LLM is available
# if llm:
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
# else:
#     qa = None

# # Flask routes
# @app.route("/")
# def index():
#     return render_template("chat.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.json.get("message")
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     if qa:
#         try:
#             result = qa.invoke({"query": user_input})
#             return jsonify({"response": result["result"]})
#         except Exception as e:
#             return jsonify({"response": f"I'm sorry, I'm having technical difficulties: {str(e)}"})
#     else:
#         # Fallback response when model isn't available
#         return jsonify({"response": "I'm currently unable to process your request. The model is not available. Please try again later."})

# @app.route("/get", methods=["GET", "POST"])
# def get_response():
#     msg = request.form["msg"]
    
#     if qa:
#         try:
#             result = qa.invoke({"query": msg})
#             return str(result["result"])
#         except Exception as e:
#             return f"I'm sorry, I'm having technical difficulties: {str(e)}"
#     else:
#         return "I'm currently unable to process your request. The model is not available. Please try again later."

# @app.route("/health")
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": llm is not None,
#         "qa_available": qa is not None
#     })

# # Run app
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)

















from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
import os

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Pinecone setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"

# Global variables
embeddings = None
docsearch = None
qa = None

def initialize_components():
    """Initialize components with error handling"""
    global embeddings, docsearch, qa
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(index_name)
        print("✓ Connected to Pinecone")
        
        # Download HuggingFace embeddings
        embeddings = download_hugging_face_embeddings()
        print("✓ Embeddings loaded")
        
        # Setup vector store
        docsearch = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace="default"
        )
        print("✓ Vector store connected")
        
        # For deployment, we'll use a simple text-based fallback instead of LLM
        # This avoids the large model file issue
        print("✓ Using retrieval-only mode (no LLM model needed)")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def get_context_response(user_input):
    """Get response using only vector search (no LLM)"""
    if not docsearch:
        return "Sorry, the system is not properly initialized."
    
    try:
        # Get relevant documents
        docs = docsearch.similarity_search(user_input, k=3)
        
        if not docs:
            return "I don't have information about that topic in my knowledge base."
        
        # Combine the most relevant context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a simple response based on retrieved context
        response = f"Based on the medical information available:\n\n{context[:800]}...\n\nPlease consult with a healthcare professional for personalized medical advice."
        
        return response
        
    except Exception as e:
        return f"Sorry, I encountered an error while searching: {str(e)}"

def get_llm_response(user_input):
    """Try to use local LLM if available, fallback to context-only"""
    model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
    
    # Check if model exists locally (for development)
    if os.path.exists(model_path):
        try:
            from langchain_community.llms import CTransformers
            
            # Setup prompt
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            llm = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    'max_new_tokens': 256,
                    'temperature': 0.8,
                    'context_length': 1024,
                    'threads': 1,
                }
            )
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            result = qa.invoke({"query": user_input})
            return result["result"]
            
        except Exception as e:
            print(f"LLM failed, using context-only: {e}")
            return get_context_response(user_input)
    else:
        # Model not available (deployment), use context-only
        return get_context_response(user_input)

# Initialize components on startup
initialization_success = initialize_components()

# Flask routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not initialization_success:
        return jsonify({"error": "System not initialized properly"}), 500
        
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = get_llm_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Sorry, I encountered an error: {str(e)}"})

@app.route("/get", methods=["GET", "POST"])
def get_response():
    if not initialization_success:
        return "System not initialized properly"
        
    msg = request.form["msg"]
    
    try:
        response = get_llm_response(msg)
        return str(response)
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

@app.route("/health")
def health():
    model_exists = os.path.exists("model/llama-2-7b-chat.ggmlv3.q4_0.bin")
    return jsonify({
        "status": "healthy" if initialization_success else "degraded",
        "pinecone_connected": docsearch is not None,
        "embeddings_loaded": embeddings is not None,
        "local_model_available": model_exists,
        "mode": "full_llm" if model_exists else "retrieval_only"
    })

# Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)