# scripts/02_build_vectorstore.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

SOURCE_FILE = "data/faq_caixa_fallback.jsonl"
DB_DIR = "db/chroma_faq"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents(path: str) -> list[Document]:
    """Carrega os dados do JSONL como documentos LangChain."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            content = f"Pergunta: {item['pergunta']}\nResposta: {item['resposta']}"
            metadata = {"tema": item["tema"]}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def build_vectorstore(docs: list[Document], db_dir: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=db_dir)
    vectordb.persist()
    return vectordb


def main():
    print("[+] Carregando documentos do FAQ...")
    raw_docs = load_faq_documents(SOURCE_FILE)
    print(f"[✓] {len(raw_docs)} documentos carregados")

    print("[+] Realizando split dos textos...")
    split_docs = split_documents(raw_docs)
    print(f"[✓] Total de chunks: {len(split_docs)}")

    print("[+] Construindo base vetorial com ChromaDB...")
    build_vectorstore(split_docs, DB_DIR)
    print(f"[✓] Vectorstore salvo em {DB_DIR}")


if __name__ == "__main__":
    main()
