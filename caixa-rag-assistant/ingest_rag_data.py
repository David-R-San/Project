# scripts/01_ingest.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DB_DIR = "db/chroma_faq"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_DIR = "data/faq"  # lugar onde os arquivos .txt estarão


def ingest_documents():
    print("[+] Carregando documentos de:", SOURCE_DIR)
    loader = DirectoryLoader(SOURCE_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print(f"[+] {len(documents)} documentos carregados.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(documents)
    print(f"[+] {len(chunks)} chunks gerados.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    vectordb.persist()
    print("[✓] Base vetorial criada com sucesso em:", DB_DIR)


if __name__ == "__main__":
    ingest_documents()
