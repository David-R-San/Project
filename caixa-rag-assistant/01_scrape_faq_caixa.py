#!/usr/bin/env python3


import requests
from bs4 import BeautifulSoup
import os
import json
from typing import List, Dict

FAQ_URLS = {
    "fgts": "https://www.caixa.gov.br/beneficios-trabalhador/fgts/perguntas-frequentes/Paginas/default.aspx",
    #"abono_salarial": "https://www.caixa.gov.br/beneficios-trabalhador/abono-salarial/perguntas-frequentes/Paginas/default.aspx",
    #"seguro_desemprego": "https://www.caixa.gov.br/beneficios-trabalhador/seguro-desemprego/perguntas-frequentes/Paginas/default.aspx",
    #"bolsa_familia": "https://www.caixa.gov.br/programas-sociais/bolsa-familia/perguntas-frequentes/Paginas/default.aspx"
}

def generic_faq_extractor(url: str, tema: str) -> List[Dict[str, str]]:
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    faqs = []

    headings = soup.find_all(['h2', 'h3', 'h4'])
    for h in headings:
        pergunta = h.get_text(strip=True)
        resposta_parts = []

        next_tag = h.find_next_sibling()
        while next_tag and next_tag.name in ['p', 'div', 'ul']:
            resposta_parts.append(next_tag.get_text(strip=True))
            next_tag = next_tag.find_next_sibling()

        if pergunta and resposta_parts:
            resposta = " ".join(resposta_parts)
            faqs.append({"tema": tema, "pergunta": pergunta, "resposta": resposta})

    print(f"[+] {tema.upper()}: {len(faqs)} itens")
    return faqs

def save_jsonl(data: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[✓] Salvo em {path}")

def main():
    all_faqs = []
    for tema, url in FAQ_URLS.items():
        try:
            faqs = generic_faq_extractor(url, tema)
            all_faqs.extend(faqs)
        except Exception as e:
            print(f"[X] Erro ao processar {tema}: {e}")

    save_jsonl(all_faqs, "data/faq_caixa_fallback.jsonl")

if __name__ == "__main__":
    main()
