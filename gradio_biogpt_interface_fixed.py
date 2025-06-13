import torch
import gradio as gr
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from models.clinical_model import ClinicalMultimodalModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
tokenizer.pad_token = tokenizer.eos_token
model = ClinicalMultimodalModel("microsoft/biogpt")
#model.load_state_dict(torch.load("model_biogpt.pth", map_location=device))
model.eval()
model.to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Grad-CAM hook storage
grads = []
acts = []

def save_grad(_, __, grad_output):
    grads.append(grad_output[0])

def save_act(_, __, output):
    acts.append(output)

# Attach hooks to final conv layer
target_layer = model.encoder.layer4[2].conv3
target_layer.register_forward_hook(save_act)
target_layer.register_backward_hook(save_grad)

def predict(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    prompt = "Findings: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device).long()

    grads.clear()
    acts.clear()
    model.zero_grad()

    # No torch.no_grad here: we want gradients from encoder
    visual_features = model.encoder(image_tensor)
    visual_embeds = model.projector(visual_features).unsqueeze(1)

    inputs_embeds = model.decoder.get_input_embeddings()(input_ids)
    combined = torch.cat([visual_embeds, inputs_embeds[:, 1:, :]], dim=1)

    attention_mask = torch.cat([
        torch.ones((input_ids.size(0), 1), device=device),
        input_ids.new_ones(input_ids.shape)[:, 1:]
    ], dim=1)

    # Loss for Grad-CAM
    loss = model.decoder(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        labels=input_ids
    ).loss
    loss.backward()

    # Grad-CAM computation
    grad = grads[0].cpu().detach().numpy()[0]
    act = acts[0].cpu().detach().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(image.resize((224, 224))).astype(np.uint8)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Generate final report
    with torch.no_grad():
        generated = model.decoder.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_length=256,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text, Image.fromarray(overlay)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="📄 Report BioGPT"), gr.Image(label="Grad-CAM")],
    title="Radiology Report Generation with BioGPT + Grad-CAM",
    description="Upload a chest X-ray image to automatically generate a report with visual attention (Grad-CAM)."
)

if __name__ == "__main__":
    iface.launch()
