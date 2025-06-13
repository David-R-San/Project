import torch
import time
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset.chex_dataset import CheXDataset
from models.clinical_model import ClinicalMultimodalModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
import pandas as pd
import os

# Setup
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model candidates
model_variants = {
    "BioGPT": "microsoft/biogpt",
    "BioGPT-Large": "microsoft/biogpt-large",
    "GPT-2": "gpt2",
   # "PMC_LLaMA_7B": "chaoyi-wu/PMC_LLAMA_7B",#precisa de mais vram
    #"MedAlpaca": "medalpaca/medalpaca-7b",
    #"LLaMA-2-7B": "NousResearch/Llama-2-7b-hf",
    #"Mistral-7B": "mistralai/Mistral-7B-v0.1", #ate aqui precia de mais vram
    "Phi-2": "microsoft/phi-2",
    #"TinyLLaMA": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Evaluation metrics
rouge = Rouge()
smooth = SmoothingFunction().method4
all_results = []

# Import CheXbert classifier (assumed available)
#from utils.chexbert_eval import compute_chexbert_accuracy

for model_name, model_checkpoint in model_variants.items():
    print(f"\n===== Evaluating {model_name} =====")

    if model_name == "PMC_LLaMA_7B":
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    dataset = CheXDataset(
        reports_csv="data/normalized_reports.csv",
        projections_csv=r"C:/Users/David/Downloads/archive (1)/indiana_projections.csv",
        image_dir=r"C:/Users/David/Downloads/archive (1)/images/images_normalized",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ClinicalMultimodalModel(model_checkpoint).to(device)
    model.eval()

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []
    rouge_scores = []
    chexbert_preds = []
    chexbert_targets = []
    inference_times = []
    examples = []

    empty_log_path = "results/empty_outputs_log.txt"
    with open(empty_log_path, "a", encoding="utf-8") as log_file:
        for idx, batch in enumerate(tqdm(dataloader)):
            image = batch["image"].to(device)
            input_ids = tokenizer.encode("Findings: ", return_tensors="pt").to(device).long()

            start_time = time.time()
            with torch.no_grad():
                visual_features = model.encoder(image)
                visual_embeds = model.projector(visual_features).unsqueeze(1)
                inputs_embeds = model.decoder.get_input_embeddings()(input_ids)
                combined = torch.cat([visual_embeds, inputs_embeds[:, 1:, :]], dim=1)
                attention_mask = torch.cat([
                    torch.ones((input_ids.size(0), 1), device=device),
                    input_ids.new_ones(input_ids.shape)[:, 1:]
                ], dim=1)

                
                '''generated = model.decoder.generate(
                    inputs_embeds=combined,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=256,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True
                )'''
                generated = model.decoder.generate(
                    input_ids=input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=256,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True
                )




            inference_times.append(time.time() - start_time)

            output_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            reference_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True).strip()

            ref = reference_text.split() or ["none"]
            hyp = output_text.split()

            if not hyp:
                print(f"[WARNING] Empty output at index {idx} for model {model_name}")
                log_file.write(f"Model: {model_name} | Index: {idx}\nReference: {reference_text}\nGenerated: (empty)\n\n")
                output_text = "none"
                bleu_1_scores.append(0.0)
                bleu_2_scores.append(0.0)
                bleu_4_scores.append(0.0)
                rouge_scores.append(0.0)
            else:
                bleu_1_scores.append(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smooth))
                bleu_2_scores.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
                bleu_4_scores.append(sentence_bleu([ref], hyp, smoothing_function=smooth))
                try:
                    rouge_scores.append(rouge.get_scores(output_text, reference_text)[0]["rouge-l"]["f"])
                except ValueError:
                    rouge_scores.append(0.0)

            chexbert_preds.append(output_text)
            chexbert_targets.append(reference_text)

            examples.append({
                "Model": model_name,
                "Generated": output_text,
                "Reference": reference_text
            })

    summary = {
        "Model": model_name,
        "BLEU-1": sum(bleu_1_scores) / len(bleu_1_scores),
        "BLEU-2": sum(bleu_2_scores) / len(bleu_2_scores),
        "BLEU-4": sum(bleu_4_scores) / len(bleu_4_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
       #"CheXbert Accuracy": compute_chexbert_accuracy(chexbert_preds, chexbert_targets),
        "Avg Inference Time (s)": sum(inference_times) / len(inference_times)
    }

    print("\n===== Evaluation Summary =====")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # Salva resultados imediatamente
    pd.DataFrame([summary]).to_csv(f"results/summary_{model_name}.csv", index=False)
    pd.DataFrame(examples).to_csv(f"results/generated_vs_reference_{model_name}.csv", index=False)

    all_results.append(summary)

print("\n✅ Resultados salvos em results")
