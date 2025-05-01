
# from langchain_community.document_loaders import PyPDFLoader
# # from langchain.document_loaders import DirectoryLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings

# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings

# def load_pdf(data):
#     loader = DirectoryLoader(
#         data,
#         glob="*.pdf",
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     return documents



#     #create text chunks
# def text_split(extracted_data):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
#   text_chunks = text_splitter.split_documents(extracted_data)

#   return text_chunks



#   #download embedding model
# def download_hugging_face_embeddings():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return embeddings












from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import

def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
