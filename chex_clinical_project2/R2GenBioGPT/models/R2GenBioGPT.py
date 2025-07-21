import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, SwinModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from peft import get_peft_model, LoraConfig, TaskType
from transformers import SwinModel
from peft.tuners.lora import Linear as LoRALinear
import torch.nn as nn
from peft import PeftModelForFeatureExtraction


torch.set_float32_matmul_precision('medium')



class R2GenBioGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder: {args.vision_model}')
        
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
      
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        print(f'Frozen vision encoder: {args.vision_model} -- Done')

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')









        print('Loading BioGPT')
        self.tokenizer = AutoTokenizer.from_pretrained(args.biogpt_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = AutoModelForCausalLM.from_pretrained(args.biogpt_model)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.embed_tokens = self.decoder.get_input_embeddings()
        print('nao descongela BioGPT decoder -- Done')



    


        

        
        


        
        #from peft import get_peft_model, LoraConfig, TaskType

        base_model = AutoModelForCausalLM.from_pretrained(args.biogpt_model)

        # Congelar tudo
        for param in base_model.parameters():
            param.requires_grad = False

        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # <-- ajuste conforme o modelo
        )
        
        # Envolve com LoRA
        self.decoder = get_peft_model(base_model, peft_config)
        print(f"[INFO] Delta Alignment (LoRA) ativado no decoder.")
        



        #descongerlar biogpt abaio
        '''
        print('Loading BioGPT')
        self.tokenizer = AutoTokenizer.from_pretrained(args.biogpt_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = AutoModelForCausalLM.from_pretrained(args.biogpt_model)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.embed_tokens = self.decoder.get_input_embeddings()
        print('descongela BioGPT decoder -- Done')
        '''

        self.proj = nn.Linear(self.visual_encoder.config.hidden_size, self.decoder.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.decoder.config.hidden_size)

        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        self.reports_csv = args.reports_csv
        self.projections_csv = args.projections_csv
        self.image_dir = args.image_dir

        self.save_gradcam_samples = 10
        self.gradcam_samples_saved = 0
        self.gradcam_dir = os.path.join(self.hparams.savedmodel_path, "gradcam")
        os.makedirs(self.gradcam_dir, exist_ok=True)

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Loaded checkpoint from {args.delta_file}')

    def on_train_start(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.train()  # <-- necessário!
        print("[INFO] Visual encoder não foi descongelado para treino profundo")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    '''
    
    def configure_optimizers(self):
        # Parâmetros treináveis
        encoder_params = list(self.visual_encoder.parameters())
        decoder_params = list(self.proj.parameters()) + list(self.layer_norm.parameters())

        # Se decoder estiver descongelado, adiciona os parâmetros dele
        if any(p.requires_grad for p in self.decoder.parameters()):
            decoder_params += list(self.decoder.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.hparams.lr_encoder},
            {"params": decoder_params, "lr": self.hparams.lr_decoder},
        ])

        # Scheduler: Reduz LR se val_loss parar de melhorar
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=2, factor=0.5, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    '''

    def encode_img(self, images):
        #image_embed = self.visual_encoder(images)['last_hidden_state'].mean(dim=1)
        image_embed = self.visual_encoder(pixel_values=images)['last_hidden_state'].mean(dim=1)

        img_embeds = self.proj(image_embed)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds = img_embeds.unsqueeze(1)
        attention_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
        return img_embeds, attention_mask

    def prompt_wrap(self, img_embeds, atts_img):
        batch_size = img_embeds.shape[0]
        prompt_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(img_embeds.device)
        prompt_embeds = self.embed_tokens(prompt_ids).expand(batch_size, -1, -1)
        wrapped_embeds = torch.cat([img_embeds, prompt_embeds], dim=1)
        wrapped_attention = torch.cat([atts_img, torch.ones_like(prompt_ids).expand(batch_size, -1)], dim=1)
        return wrapped_embeds, wrapped_attention

    def forward(self, samples):
        image = samples["image"]
        input_ids = samples["input_ids"].to(image.device)
        attention_mask = samples["attention_mask"].to(image.device)

        img_embeds, atts_img = self.encode_img(image)
        inputs_embeds, attention_mask_prefix = self.prompt_wrap(img_embeds, atts_img)

        input_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([attention_mask_prefix, attention_mask], dim=1)

        labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        labels = torch.cat([
            torch.full((labels.size(0), attention_mask_prefix.size(1)), -100, device=labels.device),
            labels
        ], dim=1)

        assert inputs_embeds.size(1) == labels.size(1)

        output = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return {"loss": output.loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log("train_loss", result["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return result


    def validation_step(self, batch, batch_idx):
        result = self(batch)
        self.log("val_loss", result["loss"], prog_bar=True, on_epoch=True)
        return result

    def on_train_epoch_end(self):
        if not hasattr(self, 'loss_log'):
            self.loss_log = []
        if not hasattr(self, 'val_loss_log'):
            self.val_loss_log = []

        train_loss = self.trainer.logged_metrics.get('train_loss')

        if train_loss is not None:
            self.loss_log.append(train_loss.item())

        val_loss = self.trainer.logged_metrics.get('val_loss')
        if val_loss is not None:
            self.val_loss_log.append(val_loss.item())

        if self.current_epoch + 1 == self.trainer.max_epochs:
            path = os.path.join(self.hparams.savedmodel_path, "loss_curve.png")
            plt.figure()
            plt.plot(range(1, len(self.loss_log)+1), self.loss_log, label='Train Loss')
            if self.val_loss_log:
                plt.plot(range(1, len(self.val_loss_log)+1), self.val_loss_log, label='Val Loss')
            plt.title("Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(path)
            plt.close()
            print(f"[INFO] Salvo gráfico de loss em {path}")

    def generate(self, samples):
        image = samples["image"]
        input_ids = samples.get("input_ids")
        attention_mask = samples.get("attention_mask")

        img_embeds, atts_img = self.encode_img(image)
        prompt_embeds, prompt_mask = self.prompt_wrap(img_embeds, atts_img)

        if input_ids is not None and attention_mask is not None:
            input_ids = input_ids.to(image.device)
            attention_mask = attention_mask.to(image.device)
            input_embeds = self.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            inputs_embeds = prompt_embeds
            attention_mask = prompt_mask

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.hparams.max_new_tokens,
            do_sample=self.hparams.do_sample,
            num_beams=self.hparams.beam_size,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            repetition_penalty=self.hparams.repetition_penalty,
            temperature=self.hparams.temperature,
            length_penalty=self.hparams.length_penalty,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def test_step(self, batch, batch_idx):
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
            self.gradcam_samples_saved = 0

        output_text = self.generate(batch)
        for i in range(len(batch["id"])):
            sample = {
                "id": batch["id"][i],
                "gt": batch["input_text"][i],
                "pred": output_text[i]
            }
            self.test_step_outputs.append(sample)
            '''
            if self.gradcam_samples_saved < self.save_gradcam_samples:
                self.save_gradcam(batch["image"][i], batch["id"][i])
                self.gradcam_samples_saved += 1
            '''    

    @torch.no_grad()
    def save_gradcam(self, image_tensor, image_id, save_dir=None):
        import os
        import torch
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt

        self.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        image_tensor.requires_grad = True  # Enable grad for input

        # Hook to capture output and retain grad
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            output.retain_grad()
            self._features = output

        # Register hook
        handle = self.visual_encoder.encoder.layers[-1].register_forward_hook(hook_fn)

        # Forward pass
        img_feat = self.visual_encoder(image_tensor)['last_hidden_state']  # Triggers hook

        handle.remove()

        # Get grad and activation
        features = self._features  # [1, N, C]
        grad = features.grad  # [1, N, C]
        weights = grad.mean(dim=1, keepdim=True)  # [1, 1, C]
        cam = (weights * features).sum(dim=-1).squeeze()  # [N] → spatial
        cam = cam.cpu().detach().numpy()

        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # Prepare image
        img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.uint8(255 * img)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save
        save_dir = save_dir or os.path.join(self.hparams.savedmodel_path, "gradcams")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{image_id}_gradcam.png")
        cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB to BGR
        print(f"[GradCAM] Saved to {save_path}")



    def on_test_end(self):
        metrics = self.compute_test_metrics()
        with open(os.path.join(self.hparams.savedmodel_path, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(self.hparams.savedmodel_path, "predictions.csv"), "w", encoding="utf-8") as f:
            f.write("uid,ground_truth,prediction\n")
            for sample in self.test_step_outputs:
                gt = sample["gt"].replace("\n", " ").replace(",", " ")
                pred = sample["pred"].replace("\n", " ").replace(",", " ")
                f.write(f"{sample['id']},{gt},{pred}\n")
        print(f"[INFO] Salvo: metrics.json, predictions.csv, gradcam/{self.save_gradcam_samples} imgs")

    def compute_test_metrics(self):
        print("[INFO] Calculando métricas de teste")
        gts, res = {}, {}
        for sample in self.test_step_outputs:
            gts[sample["id"]] = [sample["gt"]]
            res[sample["id"]] = [sample["pred"]]

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr"),
            (Rouge(), "ROUGE_L"),
        ]

        final_scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        output_path = os.path.join(self.hparams.savedmodel_path, "test_metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_scores, f, indent=4)
        return final_scores
