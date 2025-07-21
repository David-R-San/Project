

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer, BlipProcessor
from torch.utils.data import random_split, DataLoader
from dataset.chex_dataset import CheXDataset
from models.clinical_model import ClinicalMultimodalModelBLIP2StyleBioT5

# Settings
model_checkpoint = "microsoft/biogpt"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
pad_token_id = tokenizer.pad_token_id

# Dataset loading
full_dataset = CheXDataset(
    reports_csv="data/normalized_reports.csv",
    projections_csv=r"C:/Users/David/Downloads/archive (1)/indiana_projections.csv",
    image_dir=r"C:/Users/David/Downloads/archive (1)/images/images_normalized",
    tokenizer=tokenizer,
    processor=blip_processor
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
model = ClinicalMultimodalModelBLIP2StyleBioT5().to(device)
# Unfreeze visual encoder
for param in model.visual_encoder.parameters():
    param.requires_grad = True

initial_weight_decay = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=initial_weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1, verbose=True
)
loss_fn = torch.nn.CrossEntropyLoss()
save_dir = "saved_models_blipfull_teste_A"
os.makedirs(save_dir, exist_ok=True)

# Safe clamp for input_ids
def safe_clamp(tensor, vocab_size):
    return torch.clamp(tensor, min=0, max=vocab_size - 1)


'''
# Try to resume training from checkpoint
checkpoint_path = r"C:/Users/David/Downloads/chex_clinical_project/saved_models_blipfull_teste_A/checkpoint_epoch_15.pth"
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded weights only, not full checkpoint state.")
'''
        


# Training loop
loss_history = []
num_epochs = 30
val_loss_not_improving = 0
start_epoch = 30
best_val_loss = float('inf')  # fallback default

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["input_ids"] = safe_clamp(batch["input_ids"], tokenizer.vocab_size)
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
            batch["input_ids"] = safe_clamp(batch["input_ids"], tokenizer.vocab_size)
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    if avg_val_loss >= best_val_loss:
        val_loss_not_improving += 1
        optimizer.param_groups[0]['weight_decay'] *= 1.5
    else:
        val_loss_not_improving = 0
        best_val_loss = avg_val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, os.path.join(save_dir, "best_model.pth"))

    loss_history.append({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "lr": optimizer.param_groups[0]['lr'],
        "weight_decay": optimizer.param_groups[0]['weight_decay']
    })

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | WD: {optimizer.param_groups[0]['weight_decay']:.2e}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

# Save loss history
os.makedirs("results_blipfull_testeA", exist_ok=True)
pd.DataFrame(loss_history).to_csv("results_blipfull_testeA/loss_per_epoch.csv", index=False)

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
plt.savefig("results_blipfull_testeA/loss_curve.png")
plt.close()

# Print all losses per epoch
print("\n===== Losses Per Epoch =====")
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
print("-" * 33)
for entry in loss_history:
    print(f"{entry['epoch']:>5} | {entry['train_loss']:>10.4f} | {entry['val_loss']:>10.4f}")

# Evaluation
checkpoint_eval = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device)
#checkpoint_eval = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint_eval["model_state_dict"])
model.eval()

bleu_1, bleu_2, bleu_4, rouge_l = [], [], [], []
inf_times = []
predictions = []
smooth = SmoothingFunction().method4
rouge = Rouge()

for batch in tqdm(test_loader, desc="Evaluating"):
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["input_ids"] = safe_clamp(batch["input_ids"], tokenizer.vocab_size)

    start = time.time()
    with torch.no_grad():
        generated_ids = model(
            image=batch["image"],
            generate=True,
            max_length=256,
            do_sample=True,
            #top_p=0.9,
            repetition_penalty=1.2,
            length_penalty=1.0,
            num_beams=5
        )
    duration = time.time() - start
    inf_times.append(duration)

    pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip() or "none"

    input_ids = batch["input_ids"].squeeze(0) if batch["input_ids"].dim() == 2 else batch["input_ids"]
    ref_text = tokenizer.decode(input_ids, skip_special_tokens=True).strip() or "none"

    predictions.append({
        "generated": pred_text,
        "reference": ref_text,
        "inference_time_s": duration
    })

    ref, hyp = [ref_text.split()], pred_text.split()
    bleu_1.append(sentence_bleu(ref, hyp, weights=(1,0,0,0), smoothing_function=smooth))
    bleu_2.append(sentence_bleu(ref, hyp, weights=(0.5,0.5,0,0), smoothing_function=smooth))
    bleu_4.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
    rouge_l.append(rouge.get_scores(pred_text, ref_text)[0]["rouge-l"]["f"])

summary = {
    "BLEU-1": sum(bleu_1)/len(bleu_1),
    "BLEU-2": sum(bleu_2)/len(bleu_2),
    "BLEU-4": sum(bleu_4)/len(bleu_4),
    "ROUGE-L": sum(rouge_l)/len(rouge_l),
    "Avg Inference Time (s)": sum(inf_times)/len(inf_times)
}

print("\n===== Final Evaluation Results =====")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

pd.DataFrame([summary]).to_csv("results_blipfull_testeA/final_metrics.csv", index=False)
pd.DataFrame(predictions).to_csv("results_blipfull_testeA/generated_vs_reference.csv", index=False)
