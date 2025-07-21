# file: train_eval_with_metrics.py

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer
from torch.utils.data import random_split, DataLoader
from dataset.chex_dataset import CheXDataset
from models.clinical_model import ClinicalMultimodalModel

# Settings
model_checkpoint = "microsoft/biogpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# Dataset loading
full_dataset = CheXDataset(
    reports_csv="data/normalized_reports.csv",
    projections_csv=r"C:/Users/David/Downloads/archive (1)/indiana_projections.csv",
    image_dir=r"C:/Users/David/Downloads/archive (1)/images/images_normalized",
    tokenizer=tokenizer
)

# Seed for reproducibility
torch.manual_seed(42)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
model = ClinicalMultimodalModel(model_checkpoint).to(device)

checkpoint_path =r"C:/Users/David/Downloads/chex_clinical_project/saved_models/checkpoint_epoch_3_melhor.pth"

# Load checkpoint explicitly
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

initial_weight_decay = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=initial_weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=0, verbose=True
)
loss_fn = torch.nn.CrossEntropyLoss()
save_dir = "saved_modelsNeo"
os.makedirs(save_dir, exist_ok=True)

# Training loop
best_val_loss = float('inf')
loss_history = []
num_epochs = 0
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    if avg_val_loss >= best_val_loss:
        no_improve_epochs += 1
    else:
        no_improve_epochs = 0
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    if no_improve_epochs >= 1:
        new_wd = optimizer.param_groups[0]['weight_decay'] * 1.5
        optimizer.param_groups[0]['weight_decay'] = new_wd
    else:
        new_wd = optimizer.param_groups[0]['weight_decay']

    loss_history.append({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "lr": optimizer.param_groups[0]['lr'],
        "weight_decay": new_wd
    })

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | WD: {new_wd:.1e}")

    torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

# Save loss history
os.makedirs("resultsNeo", exist_ok=True)
pd.DataFrame(loss_history).to_csv("resultsNeo/loss_per_epoch.csv", index=False)

plt.figure(figsize=(10, 6))
epochs = [entry["epoch"] for entry in loss_history]
train_losses = [entry["train_loss"] for entry in loss_history]
val_losses = [entry["val_loss"] for entry in loss_history]
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("resultsNeo/loss_curve.png")
plt.close()

# Evaluation
#checkpoint_eval = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device)# modifcar  com o de baixo para usar modelo salvo
checkpoint_eval = torch.load(checkpoint_path, map_location=device)
#model.load_state_dict(checkpoint_eval["model_state_dict"])
model.eval()

bleu_1, bleu_2, bleu_4, rouge_l = [], [], [], []
inf_times, ce_losses = [], []
predictions = []
smooth = SmoothingFunction().method4
rouge = Rouge()

for batch in tqdm(test_loader, desc="Evaluating"):
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = tokenizer.encode("Findings: ", return_tensors="pt").to(device).long()

    start = time.time()
    with torch.no_grad():
        visual = model.encoder(batch["image"])
        visual_proj = model.projector(visual).unsqueeze(1)
        embed_input = model.decoder.get_input_embeddings()(input_ids)
        combined = torch.cat([visual_proj, embed_input[:, 1:, :]], dim=1)
        attention_mask = torch.cat([
            torch.ones((input_ids.size(0), 1), device=device),
            input_ids.new_ones(input_ids.shape)[:, 1:]
        ], dim=1)
        output = model.decoder.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_length=256,
            do_sample=False,
            num_beams=3
        )
    inf_times.append(time.time() - start)

    pred_text = tokenizer.decode(output[0], skip_special_tokens=True).strip() or "none"
    ref_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True).strip() or "none"

    predictions.append({"generated": pred_text, "reference": ref_text})

    ref, hyp = [ref_text.split()], pred_text.split()
    bleu_1.append(sentence_bleu(ref, hyp, weights=(1,0,0,0), smoothing_function=smooth))
    bleu_2.append(sentence_bleu(ref, hyp, weights=(0.5,0.5,0,0), smoothing_function=smooth))
    bleu_4.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
    rouge_l.append(rouge.get_scores(pred_text, ref_text)[0]["rouge-l"]["f"])

    with torch.no_grad():
        outputs = model(**batch)
        ce_losses.append(outputs.loss.item())

summary = {
    "BLEU-1": sum(bleu_1)/len(bleu_1),
    "BLEU-2": sum(bleu_2)/len(bleu_2),
    "BLEU-4": sum(bleu_4)/len(bleu_4),
    "ROUGE-L": sum(rouge_l)/len(rouge_l),
    "Perplexity": torch.exp(torch.tensor(sum(ce_losses)/len(ce_losses))).item(),
    "Avg Inference Time (s)": sum(inf_times)/len(inf_times)
}

print("\n===== Final Evaluation Results =====")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

pd.DataFrame([summary]).to_csv("resultsNeo/final_metrics.csv", index=False)
pd.DataFrame(predictions).to_csv("resultsNeo/generated_vs_reference.csv", index=False)
