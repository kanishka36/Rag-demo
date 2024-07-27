import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from pinecone import Pinecone

def load_environment_variables():
    load_dotenv("credentials.env")

def initialize_retriever():
    load_environment_variables()

    index_name = "smart-index"
    hf_token = os.getenv("HF_TOKEN")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    embeddings = HuggingFaceEmbeddings()

    pc = Pinecone(api_key=pinecone_api_key)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=hf_token)

    knowledge = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    global qa
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever()
    )

def query_retriever(query: str) -> str:
    return qa.run(query)

