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
    "BioGPT": {
        "pretrained": "microsoft/biogpt",
        "checkpoint": "C:/Users/David/Downloads/chex_clinical_project/model_biogpt.pth"
    },
    #"GPT-2": {
    #    "pretrained": "gpt2",
    #    "checkpoint": "C:/Users/David/Downloads/chex_clinical_project/saved_models/model_gpt2.pth"
    #}
}

# Evaluation metrics
rouge = Rouge()
smooth = SmoothingFunction().method4
all_results = []

# Import CheXbert classifier (assumed available)
from utils.chexbert_eval import compute_chexbert_accuracy

for model_name, model_info in model_variants.items():
    print(f"\n===== Evaluating {model_name} =====")

    tokenizer = AutoTokenizer.from_pretrained(model_info["pretrained"])
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    dataset = CheXDataset(
        reports_csv = "data/normalized_reports.csv",
        projections_csv=r"C:/Users/David/Downloads/archive (1)/indiana_projections.csv",
        image_dir=r"C:/Users/David/Downloads/archive (1)/images/images_normalized",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ClinicalMultimodalModel(model_info["pretrained"]).to(device)
    model.load_state_dict(torch.load(model_info["checkpoint"], map_location=device))
    model.eval()

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []
    rouge_scores = []
    chexbert_preds = []
    chexbert_targets = []
    inference_times = []

    for batch in tqdm(dataloader):
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

            generated = model.decoder.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_length=256,
                do_sample=False,
                num_beams=3,
                early_stopping=True
            )
        inference_times.append(time.time() - start_time)

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip() or "none"
        reference_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True).strip() or "none"

        ref = [reference_text.split()]
        hyp = output_text.split()

        bleu_1_scores.append(sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu_2_scores.append(sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu_4_scores.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
        rouge_scores.append(rouge.get_scores(output_text, reference_text)[0]["rouge-l"]["f"])

        chexbert_preds.append(output_text)
        chexbert_targets.append(reference_text)

    summary = {
        "Model": model_name,
        "BLEU-1": sum(bleu_1_scores)/len(bleu_1_scores),
        "BLEU-2": sum(bleu_2_scores)/len(bleu_2_scores),
        "BLEU-4": sum(bleu_4_scores)/len(bleu_4_scores),
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores),
        "Avg Inference Time (s)": sum(inference_times)/len(inference_times)
    }

    print("\n===== Evaluation Summary =====")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    all_results.append(summary)

pd.DataFrame(all_results).to_csv("results/evaluation_comparison.csv", index=False)
print("\n✅ Resultados salvos em results/evaluation_comparison.csv")