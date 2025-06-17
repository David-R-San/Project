# utils/mlflow_utils.py

import mlflow
import os
from datetime import datetime


def start_run(query: str, embedding_model: str, llm_model: str, retriever_k: int, score_threshold: float):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("RAG_QA_Experiment")

    run = mlflow.start_run(run_name=f"query_{datetime.now().isoformat()}")

    mlflow.log_param("query", query)
    mlflow.log_param("embedding_model", embedding_model)
    mlflow.log_param("llm_model", llm_model)
    mlflow.log_param("retriever_k", retriever_k)
    mlflow.log_param("score_threshold", score_threshold)

    return run


def log_metrics(doc_count: int, elapsed_time: float):
    mlflow.log_metric("num_documents_retrieved", doc_count)
    mlflow.log_metric("response_time_seconds", elapsed_time)


def log_result(response: str):
    mlflow.log_text(response, "response.txt")


def end_run():
    mlflow.end_run()
