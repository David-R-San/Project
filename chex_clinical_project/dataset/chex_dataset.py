
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
from transformers import GPT2Tokenizer

class CheXDataset(Dataset):
    def __init__(self, reports_csv, projections_csv, image_dir, tokenizer, max_length=512):
        self.reports_df = pd.read_csv(reports_csv)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_dir = image_dir

        # Map uid → list of filenames
        self.image_map = self._load_projections(projections_csv)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

    def _load_projections(self, projections_csv):
        df = pd.read_csv(projections_csv)
        image_map = {}
        for _, row in df.iterrows():
            uid = str(row['uid'])
            file = row['filename']
            if uid not in image_map:
                image_map[uid] = []
            image_map[uid].append(file)
        return image_map

    def __len__(self):
        return len(self.reports_df)

    def __getitem__(self, idx):
        row = self.reports_df.iloc[idx]
        uid = str(row['uid'])
        image_files = self.image_map.get(uid, [])

        if not image_files:
            raise FileNotFoundError(f"Nenhuma imagem encontrada para uid {uid}")

        # Use the first available image
        image_path = os.path.join(self.image_dir, image_files[0])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Laudo textual
        #report = str(row.get('findings', '')) + ' ' + str(row.get('impression', ''))
        report = str(row.get('report', ''))
        report = report.strip()
        if not report:
            report = "No findings reported."


        encoding = self.tokenizer(report, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }








