# metrics.py
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm

def compute_text_metrics(model, dataset, tokenizer, device):
    model.eval()
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []
    rouge_scores = []

    rouge = Rouge()
    smooth = SmoothingFunction()

    for sample in tqdm(dataset, desc="Evaluating"):
        image = sample["image"].unsqueeze(0).to(device)
        input_ids = tokenizer.encode("", return_tensors="pt").to(device)

        with torch.no_grad():
            visual_feats = model.encoder(image)
            visual_embeds = model.projector(visual_feats).unsqueeze(1)

            gen_input = torch.cat([
                visual_embeds,
                model.decoder.get_input_embeddings()(input_ids)[:, 1:, :]
            ], dim=1)

            generated = model.decoder.generate(
                inputs_embeds=gen_input,
                max_length=256,
                do_sample=False,
                num_beams=3,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        reference_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

        output_tokens = output_text.split()
        reference_tokens = [reference_text.split()]

        bleu_1_scores.append(sentence_bleu(reference_tokens, output_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth.method1))
        bleu_2_scores.append(sentence_bleu(reference_tokens, output_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1))
        bleu_4_scores.append(sentence_bleu(reference_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1))

        try:
            rouge_l = rouge.get_scores(output_text, reference_text)[0]["rouge-l"]["f"]
        except:
            rouge_l = 0.0
        rouge_scores.append(rouge_l)

    return (
        sum(bleu_1_scores) / len(bleu_1_scores),
        sum(bleu_2_scores) / len(bleu_2_scores),
        sum(bleu_4_scores) / len(bleu_4_scores),
        sum(rouge_scores) / len(rouge_scores)
    )
