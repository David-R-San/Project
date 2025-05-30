# -*- coding: utf-8 -*-
"""limpa_dados.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pYe9n7jngIISZV9brMJgxzaAMPCNVUOr
"""

import os
import pandas as pd

def clean_data(file_path, output_path):
    # Carregar o arquivo Parquet consolidado
    data = pd.read_parquet(file_path, engine="fastparquet")

    # Manter apenas as colunas desejadas
    columns_to_keep = ['P-MON-CKP', 'P-PDG', 'P-TPT', 'T-TPT', 'class', 'state']
    data = data[columns_to_keep]


    # Remover registros das classes 3 e 4
    data = data[~data['class'].isin([3, 4])]

    # Tratar valores NaN (remover registros com NaN ou preencher com a média, mediana, etc.)
    data = data.dropna()  # Aqui removemos registros com valores NaN

    # Salvar o DataFrame limpo
    data.to_parquet(output_path, engine="fastparquet")
    print(f"Dados limpos salvos em: {output_path}")

    # Exibir informações gerais da base limpa
    print(f"Total de registros após limpeza: {len(data)}")
    print("Exemplo de dados limpos:")
    print(data.head())

    return data

# Caminho do arquivo consolidado
input_file = "dados_consolidados_20percent.parquet"
output_file = "dados_limpos2.parquet"

# Limpar e salvar a base de dados
data_cleaned = clean_data(input_file, output_file)