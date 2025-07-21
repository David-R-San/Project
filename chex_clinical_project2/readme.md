# Estudo comparativo entre modelos de DNN para a construção de um encoder-decoder multimodal para geração de laudos em radiografias do tórax. 

Este repositório é dividido em duas partes principais:

## Parte 1: Treinamento Inicial dos Modelos

Os modelos foram treinados utilizando os seguintes scripts:

- `train_eval_split_metrics.py`  (resnet+biogpt)
- `train_blip.py` (blip+biot5)

Esses scripts executam as tarefas de treinamento, avaliação e separação dos dados. Certifique-se de usar as versões corretas pra cada modelo em dataset e model comentados nos arquivos `clinical_model.py` e `chex_dataset.py`.

## Parte 2: R2GenBioGPT

Dentro da pasta `R2GenBioGPT`, o treinamento foi realizado com o script:

- `train.py`

Esse script é responsável pelo treinamento do modelo R2GenBioGPT, que utiliza os dados preparados na primeira etapa. Aqui as configurações são independentes da primeira parte, mais alinhadas com as do código original do paper R2GenGPT.

## Base de Dados

Os reports (laudos) já foram tratados e estão disponíveis na pasta `dataset` deste repositório.

O restante da base de dados (imagens radiográficas) pode ser obtido pelo link:

- [Chest X-rays - Indiana University (Kaggle)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

**Importante:** certifique-se de ajustar os caminhos dos arquivos nos scripts de acordo com a estrutura local da sua máquina.

