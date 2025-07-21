import torch
from torch import nn
from transformers import BlipVisionModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput



#modelo blip+biot5
'''
class ClinicalMultimodalModelBLIP2StyleBioT5(nn.Module):
    def __init__(self, vision_model_name="Salesforce/blip-image-captioning-base",
                 decoder_model_name="QizhiPei/biot5-base"):
        super().__init__()

        self.visual_encoder = BlipVisionModel.from_pretrained(vision_model_name)
        vision_hidden_size = self.visual_encoder.config.hidden_size

        self.text_decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name)
        decoder_hidden_size = self.text_decoder.config.d_model

        if vision_hidden_size != decoder_hidden_size:
            print(f"[WARN] Projecting BLIP image output ({vision_hidden_size}) to T5 hidden size ({decoder_hidden_size})")
            self.vision_projector = nn.Linear(vision_hidden_size, decoder_hidden_size)
        else:
            self.vision_projector = nn.Identity()

        self.decoder_start_token_id = self.text_decoder.config.decoder_start_token_id
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.text_decoder.config.bos_token_id
        if self.decoder_start_token_id is None:
            raise ValueError("Decoder start token ID could not be determined from config.")

        self.pad_token_id = self.text_decoder.config.pad_token_id
        self.vocab_size = self.text_decoder.config.vocab_size

    def safe_clamp(self, tensor):
        return torch.clamp(tensor, min=0, max=self.vocab_size - 1)

    def forward(self, image, input_ids=None, attention_mask=None, labels=None, generate=False, **generate_kwargs):
        visual_outputs = self.visual_encoder(pixel_values=image)  # gradientes permitidos

        encoder_hidden_states = self.vision_projector(visual_outputs.last_hidden_state)
        encoder_hidden_states = encoder_hidden_states[:, :512]  # ✂️ Corrige o erro de size mismatch (T5 espera <= 512)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        if generate:
            input_ids = torch.full(
                (image.size(0), 1),
                self.decoder_start_token_id,
                dtype=torch.long,
                device=image.device
            )
            return self.text_decoder.generate(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                pad_token_id=self.pad_token_id,
                **generate_kwargs
            )

        if input_ids is not None:
            input_ids = self.safe_clamp(input_ids)
        if labels is not None:
            labels = self.safe_clamp(labels)

        return self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs
        )
'''




import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import AutoModelForCausalLM



#modelo resnet + biogpt
class ClinicalMultimodalModel(nn.Module):
    def __init__(self, decoder_model="microsoft/biogpt"):
        super().__init__()
        self.encoder = resnet50(weights="IMAGENET1K_V1")
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 768)

        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model)
        self.projector = nn.Linear(768, self.decoder.config.hidden_size)
        self.dropout = nn.Dropout(p=0.3)



    def forward(self, image, input_ids, attention_mask):
        #with torch.no_grad():# encoder congelado
        visual_features = self.encoder(image)
        visual_embeds = self.projector(visual_features).unsqueeze(1)

        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        combined = torch.cat((visual_embeds, inputs_embeds[:, 1:, :]), dim=1)
        attention_mask = torch.cat((torch.ones((input_ids.size(0), 1), device=input_ids.device), attention_mask[:, 1:]), dim=1)

        outputs = self.decoder(inputs_embeds=combined, attention_mask=attention_mask, labels=input_ids)
        return outputs




