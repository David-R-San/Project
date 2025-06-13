
import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import AutoModelForCausalLM

class ClinicalMultimodalModel(nn.Module):
    def __init__(self, decoder_model="microsoft/biogpt"):
        super().__init__()
        self.encoder = resnet50(weights="IMAGENET1K_V1")
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 768)

        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model)
        self.projector = nn.Linear(768, self.decoder.config.hidden_size)


    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            visual_features = self.encoder(image)
        visual_embeds = self.projector(visual_features).unsqueeze(1)

        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        combined = torch.cat((visual_embeds, inputs_embeds[:, 1:, :]), dim=1)
        attention_mask = torch.cat((torch.ones((input_ids.size(0), 1), device=input_ids.device), attention_mask[:, 1:]), dim=1)

        outputs = self.decoder(inputs_embeds=combined, attention_mask=attention_mask, labels=input_ids)
        return outputs




