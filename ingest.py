# 1. Import Library
from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import os

from dotenv import load_dotenv
 
import warnings
warnings.filterwarnings("ignore")
 
load_dotenv()
 
def build_index():
        # 2. Load JSON File from Data Directory & Create Document
        loader = DirectoryLoader(path='./data', glob='**/*.csv', recursive=True)
        Ecommerce_docs = loader.load()

        # 3. RecursiveCharacter Text Splitter
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50,
                                                            separators=["\n\n", "\n", " ", "", ".",",", ";"])
 
        recursive_tokens = recursive_splitter.split_documents(Ecommerce_docs)
 
        # 4. Create embeddings using HFEmbeddings
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 5. Create Vector Store
        faiss_store = FAISS.from_documents(documents = recursive_tokens, embedding=hf_embeddings)
 
        # persist the vector store
        faiss_store.save_local("faiss_index")
 
        print("FAISS faiss_index created successfully!")
 
if __name__ == "__main__":
        build_index()
 
