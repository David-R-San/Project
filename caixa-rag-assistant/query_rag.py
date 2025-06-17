# scripts/03_query_rag.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
import time
from utils.mlflow_utils import start_run, log_metrics, log_result, end_run

DB_DIR = "db/chroma_faq"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_REPO = "HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def build_qa_chain(vectordb: Chroma) -> RetrievalQA:
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_REPO,
        huggingfacehub_api_token=HF_API_KEY,
        temperature=0.2,
        max_new_tokens=512
    )

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6, "k": 2}
    )

    prompt_template = """
    Responda à pergunta com base apenas nas informações abaixo. 
    Se não houver contexto suficiente, diga: "Desculpe, não encontrei informação suficiente para responder com base na base de conhecimento."
    Finalize a resposta completamente. Não inicie uma explicação que não será concluída.
    Responder com todos os dados possíveis listados.
    Não inclua suposições nem generalizações.
    Liste fielmente todos os dados pessoais mencionados no contexto.
    Não omita nenhuma categoria listada.

    Contexto:
    {context}

    Pergunta:
    {question}
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template.strip())

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def main():
    print("[+] Carregando base vetorial...")
    vectordb = load_vectorstore()

    print("[+] Criando cadeia RAG com modelo na HuggingFace Endpoint...")
    qa_chain = build_qa_chain(vectordb)

    print("[✓] Sistema pronto. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        query = input("\nPergunta: ").strip()
        if query.lower() in ("sair", "exit", "quit"):
            break

        docs = qa_chain.retriever.get_relevant_documents(query)
        if not docs:
            print("\n[Resposta]:\nDesculpe, não encontrei informação suficiente na base de conhecimento para responder com segurança.")
            continue

        run = start_run(query, EMBEDDING_MODEL_NAME, HF_MODEL_REPO, retriever_k=2, score_threshold=0.7)

        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        elapsed_time = time.time() - start_time

        resposta_final = result.get("result", "").strip()
        if resposta_final.endswith(("de", "para", "com", "pela", "por", "no", "na", "e", ",")):
            resposta_final = "A resposta foi incompleta. Reformule a pergunta ou tente novamente."

        log_metrics(doc_count=len(docs), elapsed_time=elapsed_time)
        log_result(resposta_final)
        end_run()

        print("\n[Resposta]:\n" + resposta_final)


if __name__ == "__main__":
    main()
