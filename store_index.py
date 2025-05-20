

# # from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# # from langchain_community.vectorstores import Pinecone as LangchainPinecone
# # from pinecone import Pinecone as PineconeClient, ServerlessSpec
# # from dotenv import load_dotenv
# # import os

# # # Load environment variables
# # load_dotenv()
# # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# # PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # e.g., aws-us-west-2
# # CLOUD, REGION = PINECONE_API_ENV.split('-')[0], '-'.join(PINECONE_API_ENV.split('-')[1:])

# # # Step 1: Load and process PDF data
# # extracted_data = load_pdf("data/")
# # text_chunks = text_split(extracted_data)
# # embeddings = download_hugging_face_embeddings()

# # # Step 2: Initialize Pinecone client
# # pc = PineconeClient(api_key=PINECONE_API_KEY)

# # # Step 3: Create index if it doesn't exist
# # index_name = "medical-chatbot"
# # if index_name not in pc.list_indexes().names():
# #     pc.create_index(
# #         name=index_name,
# #         dimension=384,  # for sentence-transformers/all-MiniLM-L6-v2
# #         metric="cosine",
# #         spec=ServerlessSpec(cloud=CLOUD, region=REGION)
# #     )

# # # Step 4: Push data to Pinecone
# # index = pc.Index(index_name)  # Get the index object

# # docsearch = LangchainPinecone.from_texts(
# #     [t.page_content for t in text_chunks],
# #     embedding=embeddings,
# #     index_name=index_name,
# #     namespace="default",  # optional
# #     pinecone_api_key=PINECONE_API_KEY
# # )








# from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # e.g., aws-us-west-2
# CLOUD, REGION = PINECONE_API_ENV.split('-')[0], '-'.join(PINECONE_API_ENV.split('-')[1:])

# # Step 1: Load and process PDF data
# extracted_data = load_pdf("data/")
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()

# # Step 2: Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Step 3: Create index if it doesn't exist
# index_name = "chatbot"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=384,  # for sentence-transformers/all-MiniLM-L6-v2
#         metric="cosine",
#         spec=ServerlessSpec(cloud=CLOUD, region=REGION)
#     )

# # Step 4: Push data to Pinecone
# index = pc.Index(index_name)  # Get the index object

# docsearch = LangchainPinecone.from_texts(
#     [t.page_content for t in text_chunks],
#     embedding=embeddings,
#     index_name=index_name,
#     namespace="default",  # optional
#     pinecone_api_key=PINECONE_API_KEY
# )


















from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


index_name="medical-bot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)