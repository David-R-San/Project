import re
import pandas as pd

def clean_xxxx(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\b[xX]{3,}\b', '', text)

def normalize_text(text):
    text = clean_xxxx(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([?.!,:;])', r'\1', text)
    text = re.sub(r'([?.!,:;])([^\s])', r'\1 \2', text)
    return text[0].upper() + text[1:] if text else text

def normalize_reports(input_path, output_path):
    df = pd.read_csv(input_path)

    # Limpa todos os campos de texto que podem conter "XXXX"
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_xxxx)

    # Constrói a nova coluna "report"
    text_columns = ["findings", "impression"]
    df["report"] = df.apply(
        lambda row: " ".join(str(row[col]) for col in text_columns if pd.notna(row[col])),
        axis=1
    )
    df["report"] = df["report"].apply(normalize_text)

    df.to_csv(output_path, index=False)
    print(f"Laudos normalizados salvos em {output_path}")
