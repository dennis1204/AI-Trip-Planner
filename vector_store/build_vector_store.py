import pickle
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

with open('hk_documents.pkl', 'rb') as f:
    documents = pickle.load(f)
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small", api_version="2023-05-15")
vector_store = FAISS.from_documents(documents, embeddings)
vector_store.save_local("hk_tourism_index")
print("Vector store built!")