# Inference — step by step

This document walks through exactly what happens when you run `scripts/test_control_lora_flux.py` or call the pipeline from your own code. Every line maps to a stage in the pipeline.

---

## 1. Load the base transformer

```python
transformer = FluxTransformer2DModel.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

This downloads the standard FLUX.1-dev transformer weights (~24 GB). At this point it has **64 input channels** — it can only accept noise, not image conditioning.

---

## 2. Expand the input layer

```python
with torch.no_grad():
    initial_input_channels = transformer.config.in_channels  # 64

    new_linear = torch.nn.Linear(
        transformer.x_embedder.in_features * 4,   # 256
        transformer.x_embedder.out_features,       # hidden_dim
        bias=transformer.x_embedder.bias is not None,
        dtype=transformer.dtype,
        device=transformer.device,
    )
    new_linear.weight.zero_()
    new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
    if transformer.x_embedder.bias is not None:
        new_linear.bias.copy_(transformer.x_embedder.bias)

    transformer.x_embedder = new_linear
    transformer.register_to_config(in_channels=initial_input_channels * 4)
```

What this does:

- Creates a new linear layer with 4x the input features (64 → 256).
- Zero-initializes **all** weights in the new layer.
- Copies the original 64-channel weights into the first 64 columns.
- The remaining 192 columns start at zero — they have no effect until the LoRA kicks in.

After this, `transformer.config.in_channels` is **256**.

---

## 3. Build the pipeline

```python
pipe = FluxControlRemovalPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")
```

`from_pretrained` loads the VAE, text encoders (CLIP + T5), tokenizers, and scheduler from the FLUX.1-dev repo. We pass in our modified transformer so it doesn't load the original one.

The pipeline object now holds:

```
pipe.transformer   → FluxTransformer2DModel (256 input channels, bf16, on GPU)
pipe.vae           → AutoencoderKL (float32)
pipe.text_encoder  → CLIPTextModel
pipe.text_encoder_2→ T5EncoderModel
pipe.tokenizer     → CLIPTokenizer
pipe.tokenizer_2   → T5TokenizerFast
pipe.scheduler     → FlowMatchEulerDiscreteScheduler
```

---

## 4. Load the LoRA weights

```python
pipe.load_lora_weights(
    'theSure/Omnieraser',
    weight_name="pytorch_lora_weights.safetensors",
)
```

This downloads the LoRA adapter (~150 MB) and merges it into the transformer's target modules. The adapter was trained to use the extra 192 input channels (masked image, foreground, mask) for object removal.

---

## 5. Prepare inputs

```python
size = (1024, 1024)
image = load_image(image_path).convert("RGB").resize(size)
mask  = load_image(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)
```

- Image must be RGB. Any resolution works, but 1024x1024 matches what the model was trained on.
- Mask is RGB here but internally gets converted to grayscale and binarized. White = remove, black = keep.
- The generator provides a fixed seed for reproducible results.

---

## 6. Call the pipeline

```python
result = pipe(
    prompt="There is nothing here.",
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=generator,
    max_sequence_length=512,
    height=1024,
    width=1024,
).images[0]
```

Here's what happens inside `pipe.__call__()`, step by step:

### 6a. Input validation

```
check_inputs() verifies:
  - height and width are divisible by vae_scale_factor * 2 = 16
  - prompt and prompt_embeds are not both provided
  - max_sequence_length <= 512
```

### 6b. Text encoding

```
encode_prompt() runs:
  CLIP tokenizer  → token_ids → CLIPTextModel   → pooled_prompt_embeds (1, 768)
  T5 tokenizer    → token_ids → T5EncoderModel  → prompt_embeds (1, 512, 4096)
  text_ids = zeros(512, 3)  ← positional IDs for text tokens
```

### 6c. Image + mask encoding

```
prepare_image_with_mask() runs:
  1. Preprocess image to tensor, normalize to [-1, 1]           → (1, 3, 1024, 1024)
  2. Preprocess mask, binarize                                   → (1, 1, 1024, 1024)
  3. Create masked_image: image with mask regions set to -1      → (1, 3, 1024, 1024)
  4. Create foreground: image with non-mask regions set to -1    → (1, 3, 1024, 1024)
  5. VAE encode masked_image                                     → (1, 16, 128, 128)
  6. VAE encode foreground                                       → (1, 16, 128, 128)
  7. Downsample mask to latent size, invert, repeat to 16ch      → (1, 16, 128, 128)
  8. Concatenate along channels                                  → (1, 48, 128, 128)
  9. Pack into 2x2 patches                                       → (1, 4096, 192)
```

### 6d. Noise initialization

```
prepare_latents() runs:
  1. Generate random noise                    → (1, 16, 128, 128)
  2. Pack into 2x2 patches                   → (1, 4096, 64)
  3. Create latent_image_ids for RoPE        → (4096, 3)
```

### 6e. Timestep schedule

```
sigmas = linspace(1.0, 1/28, 28)           → 28 evenly spaced values
mu = calculate_shift(4096)                  → adjusts for 1024x1024 resolution
timesteps = scheduler.set_timesteps(sigmas, mu=mu)
```

### 6f. Denoising loop (28 iterations)

At each step `i` with timestep `t`:

```
  1. Concatenate:  hidden_states = cat([latents, control_image], dim=2)
                                          ↓
                   shape: (1, 4096, 64) + (1, 4096, 192) = (1, 4096, 256)

  2. Predict noise:
     noise_pred = transformer(
         hidden_states = (1, 4096, 256),
         timestep      = t / 1000,
         guidance      = 3.5,
         pooled_projections    = (1, 768),
         encoder_hidden_states = (1, 512, 4096),
         txt_ids       = (512, 3),
         img_ids       = (4096, 3),
     )  →  noise_pred shape: (1, 4096, 64)

  3. Scheduler step:
     latents = scheduler.step(noise_pred, t, latents)
     → updates latents toward the clean image
```

The `control_image` tensor (192 channels) stays the **same** every step — it's the conditioning signal. Only the `latents` (64 channels) evolve through the denoising process.

### 6g. Decode

```
  1. Unpack latents: (1, 4096, 64) → (1, 16, 128, 128)
  2. Undo VAE scaling: latents = (latents / scaling_factor) + shift_factor
  3. VAE decode:       (1, 16, 128, 128) → (1, 3, 1024, 1024)
  4. Post-process to PIL image
```

### 6h. Output

```python
result = pipe(...).images[0]  # PIL.Image, 1024x1024 RGB
result.save('flux_inpaint.png')
```

---

## 7. Complete inference flow (ASCII)

```
  image (1024x1024) ─────────────┐
  mask  (1024x1024) ─────────────┤
                                  │
                           ┌──────v──────┐
                           │ Preprocess   │
                           │ & VAE encode │
                           └──────┬───────┘
                                  │
                  ┌───────────────v───────────────┐
                  │ control_image (1, 4096, 192)   │  (stays fixed)
                  └───────────────┬────────────────┘
                                  │
  random noise ──────┐            │
                     │            │
              ┌──────v──────┐     │       prompt ──── Text Encoders
              │  latents     │     │                       │
              │ (1,4096,64) │     │                       │
              └──────┬───────┘     │                       │
                     │            │                       │
         ┌───────────v────────────v───────────────────────v──┐
         │                                                    │
         │  for step = 1 to 28:                               │
         │    input = cat(latents, control_image)  (256ch)    │
         │    noise_pred = transformer(input, text, t)        │
         │    latents = scheduler.step(noise_pred, t, latents)│
         │                                                    │
         └───────────────────────┬────────────────────────────┘
                                 │
                          ┌──────v───────┐
                          │  VAE decode   │
                          └──────┬────────┘
                                 │
                          ┌──────v───────┐
                          │ Clean image   │
                          │ (1024x1024)   │
                          └───────────────┘
```

---

## 8. Key parameters and what they control

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_inference_steps` | 28 | More steps = higher quality but slower. 28 is the sweet spot. |
| `guidance_scale` | 3.5 | How strongly the text prompt influences the output. For removal, low is fine since the image conditioning dominates. |
| `max_sequence_length` | 512 | Max T5 token length. 512 is the model maximum. |
| `height`, `width` | 1024 | Output resolution. Must be divisible by 16. Trained at 1024x1024. |
| `generator` | None | Torch generator for reproducible noise. Set a seed for consistent results. |

---

## 9. Memory requirements

| Component | VRAM (approx) |
|-----------|---------------|
| FLUX transformer (bf16) | ~12 GB |
| VAE (float32) | ~160 MB |
| CLIP text encoder | ~240 MB |
| T5-XXL text encoder | ~8 GB |
| LoRA adapter | ~150 MB on disk, negligible extra VRAM |
| **Total** | **~21 GB** |

If you're running on a GPU with less VRAM, options:
- Use `pipe.enable_model_cpu_offload()` to move models to CPU when not in use.
- Use `pipe.enable_vae_tiling()` to decode large images in tiles.
- Use `pipe.enable_vae_slicing()` to decode batches one slice at a time.
