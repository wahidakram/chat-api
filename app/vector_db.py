import json
import os
from typing import List
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import config
from langchain_community.vectorstores.faiss import FAISS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",
                                   model_kwargs={"trust_remote_code": True})


def create_and_persist_embeddings(documents: List[Document]) -> FAISS:
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise e


def update_vector_store(documents: List[Document]):
    try:
        print("Updating vector store...")
        print(f'lenght of documents: {len(documents)}')
        new_documents = create_and_persist_embeddings(documents)
        # if FAISS_DB_FILE exists:
        if os.path.exists(config.FAISS_DB_FILE):
            vector_store = FAISS.load_local(config.FAISS_DB_FILE, embeddings, allow_dangerous_deserialization=True)
            print(type(vector_store))
            print(type(new_documents))
            vector_store.add_documents(documents)
            vector_store.save_local(config.FAISS_DB_FILE)
        else:
            new_documents.save_local(config.FAISS_DB_FILE)
    except Exception as e:
        print(f"Error updating vector store: {str(e)}")
        raise e


async def load_vector_store() -> FAISS:
    try:
        vector_store = FAISS.load_local(config.FAISS_DB_FILE, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        raise e
