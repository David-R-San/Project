# Comparative Study of DNN-Based Models for Building a Multimodal Encoder-Decoder for Chest X-Ray Report Generation. 

This repository is divided into two main parts

Link to the article in development: https://www.overleaf.com/read/kxqvqwmgfbjk#066db9

## Part 1: Initial Model Training

The models were trained using the following scripts:

- `train_eval_split_metrics.py`  (resnet+biogpt)
- `train_blip.py` (blip+biot5)

These scripts perform training, evaluation, and dataset splitting tasks. Make sure to use the correct versions for each model and dataset, as specified in the clinical_model.py and chex_dataset.py files.

## Part 2: R2GenBioGPT

Within the R2GenBioGPT folder, training was conducted using the script:

- `train.py`

This script is responsible for training the R2GenBioGPT model, which uses the preprocessed data from the first stage. The configurations here are independent of the first part and more closely aligned with the original codebase from the R2GenGPT paper.

## Dataset

The reports have already been preprocessed and are available in the dataset folder within this repository.

The remaining dataset (radiographic images) can be obtained from the following link:

- [Chest X-rays - Indiana University (Kaggle)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

**Important:** Be sure to adjust the file paths in the scripts according to your local machine's directory structure.

License
MIT



