# README: Comparative Study of Generative Models for Facial Image Synthesis

##  Overview
This project presents a comparative study between two Generative Adversarial Network (GAN) architectures:
- **Progressive Growing GAN (PGGAN)**
- **StyleSwin GAN**

The models were trained to generate synthetic facial images from the CelebA dataset. A detailed analysis is provided on architecture design, training behavior, output quality, and performance metrics such as FID (Fréchet Inception Distance).

---

##  Dataset and Preprocessing
- **Dataset**: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) – 202,599 images with 40 facial attributes.
- **Preprocessing**:
  - Used **MTCNN** for face detection and cropping.
  - Normalized images to \[-1, 1\] and resized progressively from 4×4 to 128×128.
  - A subset of **50,000 cleaned face images** was used for both models.

---

## Architectures

### PGGAN (Progressive Growing GAN)
- **Reference**: [Karras et al. 2018](https://arxiv.org/abs/1710.10196)
- **Framework**: Implemented in PyTorch based on a ported version of the original Keras code.
- **Key Characteristics**:
  - Starts from a low-resolution (4×4) image and progressively grows the network.
  - New layers are added gradually (fade-in) to improve stability.
  - Generator: uses upsampling + convolutional blocks.
  - Discriminator: uses downsampling blocks.
- **Hyperparameter tuning**:
  - Attempted automated tuning using **Optuna**.
  - Best performance achieved via manual tuning based on the paper.

   ## Generator – `ProgressiveGenerator`
```python
class ProgressiveGenerator(nn.Module):
    def __init__(self, latent_dim=512, depth=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.depth = depth

        # Initial block: linear transformation from latent vector to 4×4 feature map
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            PixelNormalization(),  # normalize features to stabilize training
            Conv2dWS(128, 128, 3, padding=1),
            PixelNormalization(),
            nn.LeakyReLU(0.2),
            Conv2dWS(128, 128, 3, padding=1),
            PixelNormalization(),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        for _ in range(depth):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),  # double spatial resolution
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2),
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2)
            )
            self.blocks.append(block)
            self.to_rgb.append(Conv2dWS(128, 3, 1))  # 1x1 conv to RGB output
```

## Discriminator – `ProgressiveDiscriminator`
```python
class ProgressiveDiscriminator(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()

        # Final discriminator block: adds minibatch standard deviation, then 1x1 conv to output
        self.final_conv = nn.Sequential(
            MinibatchStdev(),  # adds 1 channel with per-pixel std across minibatch
            Conv2dWS(128 + 1, 1, 1)  # output single-channel real/fake score
        )

        # Add block layers (reverse order)
        self._add_block(128, 128)
        for _ in range(1, n_blocks):
            self._add_block(128, 128)

    def _add_block(self, in_ch, out_ch):
        block = nn.Sequential(
            Conv2dWS(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2dWS(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)  # downsample spatial resolution by half
        )
        self.blocks.insert(0, block)
        self.from_rgb.insert(0, Conv2dWS(3, in_ch, 1))  # 1x1 conv to transform RGB to features
```

## Regularization Techniques

<pre>
| Technique                     | Code Implementation                                               |
|------------------------------|-------------------------------------------------------------------|
| Pixel Normalization          | `PixelNormalization` after convolutions in the  generator          |
| Minibatch StdDev             | `MinibatchStdev()` in the final discriminator block               |
| R1 Gradient Penalty          | `compute_r1_penalty()` applied to real images                     |
| Gradient Clipping            | `clip_grad_value_` on optimizer gradients                         |
| Exponential Moving Average   | `ExponentialMovingAverage()` for the generator weights            |
| Fade-in Transition           | Controlled via `alpha` blending RGB outputs during transitions    |
| Path Length Regularization   | Applied when `step > 1` using noise + gradient consistency        |
</pre>


## hyperparameters:
| Parametre           | Value                                |
| ------------------- | ------------------------------------ |
| `latent_dim`        | `512`                                |
| `lr_Generator`      | `0.0005` (Adam)                      |
| `lr_Discriminator`  | `0.00005` (Adam)                     |
| `betas` (Adam)      | `(0.0, 0.99)`                        |
| `n_batch` for step  | `[16, 16, 16, 8, 4, 4]`              |
| `n_epochs` for step | `[5, 8, 8, 10, 10, 10]`              |
| `start_step`        | `0`                                  |
| `ema_decay`         | `0.999`                              |
| `r1_weight`         | `0.01`                               |
| `grad_clip_value`   | `0.1` (used in `.clip_grad_value_`) |











### StyleSwin GAN
- **Reference**: [StyleSwin GitHub (Microsoft)](https://github.com/microsoft/StyleSwin)
- **Framework**: Original PyTorch implementation.
- **Key Characteristics**:
  - Hybrid design with **style modulation** and **Swin Transformer blocks**.
  - Fixed resolution stages (no progressive growth).
  - Generator uses hierarchical attention with shifted windows.
  - Discriminator integrates patch-based contrastive learning.
  - Modulates style at each resolution level.
- **Hyperparameters**:
  - Default settings from the official repo.
  - Modified only `max_iter`, `save_freq`, and `eval_freq` to speed up training.

 




## StyleSwin Generator and Discriminator

```python
# generator.py (StyleSwin)
class Generator(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, ...):
        # Style Encoder: latent input processed through normalization and MLP
        layers = [PixelNorm()]  # normalize latent code
        for _ in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, ...))  # project style
        self.style = nn.Sequential(*layers)

        # Constant learned 4x4 input tensor
        self.input = ConstantInput(in_channels[0])

        # Main network blocks and ToRGB conversion
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            # Main Swin Transformer block with style injection and optional upsample
            layer = StyleBasicLayer(
                dim=in_channel,
                input_resolution=(2 ** i_layer, 2 ** i_layer),
                ...
            )
            self.layers.append(layer)

            # RGB conversion after each resolution stage
            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            to_rgb = ToRGB(out_dim, upsample=(i_layer < end), resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
```

```python
# discriminator.py (StyleSwin)
class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False):
        super().__init__()

        # Channel configuration for each resolution level
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # Wavelet-based preprocessing
        self.dwt = HaarTransform(3)

        # Convolutional feature extractors
        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        # Build discriminator blocks from high to low resolution
        log_size = int(math.log(size, 2)) - 1
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))
            in_channel = out_channel

        # Final processing layers
        self.from_rgbs.append(FromRGB(channels[4], sn=sn))
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
```


## Regularization Techniques Used (StyleSwin)
<pre>
| Technique                     | Code Implementation                                                       |
|------------------------------|----------------------------------------------------------------------------|
| Dropout                      | Used in Swin Transformer blocks and attention modules                      |
| Attention Masking            | Shifted window-based self-attention ensures locality                       |
| Weight Decay                 | Applied via AdamW optimizer (standard in transformer training)             |
| LayerNorm                    | Applied before MLP and attention sub-blocks                                |
| Data Augmentation            | Includes resize, flip, crop, color jitter (via default training pipeline) |
</pre>

## Hyperparameters Used (StyleSwin)
<pre>
| Parameter              | Value                    |
|------------------------|--------------------------|
| Optimizer              | AdamW                    |
| Learning Rate          | 2e-4                     |
| Weight Decay           | 0.05                     |
| Batch Size             | 4                        |
| Training Resolution    | 128×128                  |
| Iterations             | 80k                      |
| Save Freq / Eval Freq  | 10k                      |
</pre>

**Generator Overview:**
- Starts from a fixed 4×4 learnable tensor.
- Style vector modulates blocks using MLP (EqualLinear).
- Attention blocks (StyleBasicLayer) apply Swin-style self-attention.
- Bilinear upsampling and positional encoding preserve spatial structure.
- ToRGB maps features to RGB space.

**Discriminator Overview:**
- Input processed with HaarTransform for hierarchical frequency decomposition.
- Uses convolution and SwinDiscriminatorBlock layers.
- Minibatch statistics improve discrimination.
- Final EqualLinear layer outputs real/fake prediction.



---

## Hardware Setup
- CPU: AMD Ryzen 5 5500
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 3060 (12 GB VRAM)

---

## Results Summary
| Metric        | PGGAN (32×32, step 3) | StyleSwin (128×128) |
|---------------|------------------------|---------------------|
| Max Resolution| 32×32 (step 3 only)    | 128×128             |
| Training Stability | Low (failed at step 4, 64×64) | High stability       |
| FID Score     | 330–599                | ~38                 |
| Visual Quality| Noticeable artifacts   | High fidelity        |

> **Note**: PGGAN failed to transition properly to step 4 (64×64), resulting in invalid or non-evaluable outputs (null FID). Metrics reported for PGGAN correspond to step 3 (32×32).

---

## Observations
- PGGAN struggled with transition phases, requiring careful tuning. Training instability was especially pronounced during the transition from step 3 to step 4 (64×64), which could potentially be mitigated through improved fade-in scheduling or hyperparameter tuning. However, due to persistent degradation at this stage, the model was replaced by a more modern and stable architecture.
- StyleSwin performed consistently across training steps with better global coherence due to attention mechanisms.

---

## Future Work
- Extend output resolution to 256×256 or higher.
- Explore **fairness-aware training** to address dataset bias.
- Investigate improved preprocessing pipelines (e.g., facial alignment).
- Incorporate **StyleGAN3** or **Diffusion models** for baseline comparison.
- Optimize training with more robust hyperparameter search.
- Explore the generalization capability of the StyleSwin architecture by applying it to other domains, such as medical image synthesis.


---

## License
MIT License

---

## Author
David Rocha de Santana  
Department of Computer Science  
Universidade Federal Fluminense  
[davidrs@id.uff.br](mailto:davidrs@id.uff.br)
