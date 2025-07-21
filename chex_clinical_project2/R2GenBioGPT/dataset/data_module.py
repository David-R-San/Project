import os
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')


class FieldParser:
    def __init__(self, args):
        self.args = args
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.biogpt_model)
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        if not isinstance(report, str):
            report = ""
        report = report.replace('\n', ' ').replace('..', '.').replace('  ', ' ')
        return report.strip().lower()

    def parse(self, image_path, report, uid):
        cleaned = self.clean_report(report)

        tokens = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        )

        image_full_path = os.path.join(self.args.image_dir, image_path)
        if not os.path.exists(image_full_path):
            print(f"[Warning] Image not found: {image_full_path}")
        #else:
         #  print(f"[Info] Loading image: {image_full_path}")

        with Image.open(image_full_path) as pil:
            array = np.array(pil.convert("RGB"), dtype=np.uint8)
            image = self._parse_image(array)

        return {
            "id": uid,
            "input_text": cleaned,
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "image": [image]
        }


class CustomDataset(data.Dataset):
    def __init__(self, df, parser):
        self.df = df
        self.parser = parser

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.parser.parse(row['image_path'], row['report'], row['id'])

    def collate_fn(self, batch):
        return {
            "id": [b["id"] for b in batch],
            "image": torch.stack([b["image"][0] for b in batch]),
            "input_text": [b["input_text"] for b in batch],
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch])
        }


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parser = FieldParser(args)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        df_reports = pd.read_csv(self.args.reports_csv)
        df_proj = pd.read_csv(self.args.projections_csv)

        df_proj = df_proj.rename(columns={"uid": "id"})
        df_reports = df_reports.rename(columns={"uid": "id"})
        df = pd.merge(df_proj, df_reports, on="id")
        df = df.rename(columns={"filename": "image_path"})

        if 'split' not in df.columns:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            total = len(df)
            train_end = int(0.8 * total)
            val_end = int(0.9 * total)
            df.loc[:train_end, 'split'] = 'train'
            df.loc[train_end:val_end, 'split'] = 'val'
            df.loc[val_end:, 'split'] = 'test'

        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        print(f"[DEBUG] Total imagens de treino: {len(train_df)}")

        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)

        self.dataset = {
            "train": CustomDataset(train_df, self.parser),
            "validation": CustomDataset(val_df, self.parser),
            "test": CustomDataset(test_df, self.parser)
        }

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.dataset["train"].collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.args.val_batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.dataset["validation"].collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.test_batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.dataset["test"].collate_fn,
            persistent_workers=True
        )
