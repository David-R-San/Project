
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.chex_dataset import CheXDataset
from models.clinical_model import ClinicalMultimodalModel
from tqdm import tqdm

# Escolha o modelo base para fine-tuning
model_checkpoint = "microsoft/biogpt"
#model_checkpoint = "microsoft/biogpt-large" muito grande para a gpu
#model_checkpoint = "gpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializa o tokenizador adequado ao modelo escolhido
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ajuste necessário para gpt2, pois ele não tem pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Carrega o dataset com o tokenizador configurado
dataset = CheXDataset(
    reports_csv="data/normalized_reports.csv",
    # reports_csv=r"C:\Users\David\Downloads\archive (1)\indiana_reports.csv",
    projections_csv=r"C:\Users\David\Downloads\archive (1)\indiana_projections.csv",
    image_dir=r"C:\Users\David\Downloads\archive (1)\images\images_normalized",
    tokenizer=tokenizer
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Inicializa o modelo multimodal com o checkpoint desejado
model = ClinicalMultimodalModel(model_checkpoint).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Diretório para salvar modelos
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Loop de treino
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{model_checkpoint.split('/')[-1]}_epoch_{epoch+1}.pth"))

# Salva o modelo final
torch.save(model.state_dict(), os.path.join(save_dir, f"model_{model_checkpoint.split('/')[-1]}.pth"))
