# Architecture

A technical breakdown of every component in the object removal pipeline — what it is, what shape data flows through it, and how the pieces connect.

## Method overview (from the paper)

![Method Overview](../assets/diagrams/method_overview.png)

The diagram above shows the full architecture: the input image and mask are encoded into latent space, concatenated with the foreground and noise, then processed by the FLUX transformer with LoRA adapters to produce a clean output.

---

## 1. High-level overview

The system has five major components wired together:

```
                    ┌───────────────────────┐
                    │  CLIP Text Encoder     │
  Text prompt ──────┤  (openai/clip-vit-     │──── pooled_prompt_embeds  (B, 768)
                    │   large-patch14)       │
                    └───────────────────────┘
                    ┌───────────────────────┐
  Text prompt ──────┤  T5 Text Encoder       │──── prompt_embeds  (B, seq_len, 4096)
                    │  (google/t5-v1_1-xxl)  │
                    └───────────────────────┘

                    ┌───────────────────────┐
  Input image ──────┤                        │──── image_latents  (B, 16, H/8, W/8)
  Mask ─────────────┤  VAE Encoder           │──── masked_image_latents
  Foreground ───────┤  (AutoencoderKL)       │──── foreground_latents
                    └───────────────────────┘     mask_latents  (B, 16, H/8, W/8)

                    ┌───────────────────────┐
  All latents ──────┤  FLUX Transformer      │──── denoised latents (B, 16, H/8, W/8)
  + text embeds ────┤  (FluxTransformer2D    │
  + timestep ───────┤   + LoRA adapter)      │
                    └───────────────────────┘

                    ┌───────────────────────┐
  Clean latents ────┤  VAE Decoder           │──── output image  (B, 3, H, W)
                    │  (AutoencoderKL)       │
                    └───────────────────────┘
```

---

## 2. VAE — AutoencoderKL

The VAE compresses full-resolution images into a smaller latent space and decodes them back.

| Property | Value |
|----------|-------|
| Type | `AutoencoderKL` from `diffusers` |
| Latent channels | 16 |
| Scale factor | 8 (each spatial dim shrinks by 8x) |
| Shift factor | Read from `vae.config.shift_factor` |
| Scaling factor | Read from `vae.config.scaling_factor` |
| Precision | Always kept in `float32` (even when the rest of the model is bf16) |

### Encoding formula

```python
raw_latents = vae.encode(image).latent_dist.sample()
latents = (raw_latents - shift_factor) * scaling_factor
```

### Decoding formula

```python
latents = (latents / scaling_factor) + shift_factor
image = vae.decode(latents)
```

### What gets encoded at inference

Three separate things go through the VAE encoder:

1. **Masked image** — the input photo with mask regions set to -1 (pixel value)
2. **Foreground image** — the input photo with background regions set to -1
3. **Mask** — binary mask downsampled to latent resolution, then inverted (0 = inpaint, 1 = keep), repeated across 16 channels

These are concatenated along the channel dimension:

```
control_image = cat([masked_image_latents,    # (B, 16, h, w)
                     foreground_image_latents, # (B, 16, h, w)
                     mask_latents],            # (B, 16, h, w)
                    dim=1)                     # (B, 48, h, w)
```

---

## 3. FLUX Transformer — FluxTransformer2DModel

This is the core denoising model. It's a Multimodal Diffusion Transformer (MMDiT) from Black Forest Labs.

### Standard FLUX.1-dev configuration

| Property | Value |
|----------|-------|
| Input channels | 64 (in the original model) |
| Architecture | MMDiT with joint attention between image and text tokens |
| Positional encoding | RoPE (Rotary Position Embeddings) via `latent_image_ids` |
| Guidance | Embedded as a scalar via `guidance_embeds` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (flow matching, not DDPM) |

### Input channel expansion (the key modification)

The standard FLUX transformer has an `x_embedder` linear layer that maps 64 input channels to the model's hidden dimension. For object removal, we need to feed in **4 things**: noise latents (16ch) + masked image (16ch) + foreground (16ch) + mask (16ch) = 64 channels of conditioning, plus the 64 channels of noise.

So we expand the input from 64 to **256 channels** (4x):

```python
initial_input_channels = transformer.config.in_channels  # 64

new_linear = torch.nn.Linear(
    transformer.x_embedder.in_features * 4,  # 64*4 = 256
    transformer.x_embedder.out_features,
    bias=transformer.x_embedder.bias is not None,
)

# Zero-init new weights, copy original weights into the first 64 channels
new_linear.weight.zero_()
new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)

transformer.x_embedder = new_linear
transformer.register_to_config(in_channels=initial_input_channels * 4)
```

This is a zero-initialized expansion — the extra 192 channels start at zero so the model behaves identically to the original at initialization, and the LoRA training teaches it what to do with the extra channels.

### Latent packing

FLUX uses a patch-based approach where latents are rearranged into 2x2 patches:

```python
# Pack: (B, C, H, W) -> (B, H/2 * W/2, C*4)
latents = latents.view(B, C, H//2, 2, W//2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)
latents = latents.reshape(B, (H//2)*(W//2), C*4)

# At inference, noise latents are (B, num_patches, 64)
# Control image is       (B, num_patches, 192)
# Concatenated input is  (B, num_patches, 256)
```

### What goes into the transformer at each denoising step

```python
latent_model_input = cat([latents, control_image], dim=2)  # (B, patches, 256)

noise_pred = transformer(
    hidden_states=latent_model_input,  # (B, patches, 256)
    timestep=t / 1000,                 # scalar
    guidance=guidance_scale,           # scalar (3.5)
    pooled_projections=pooled_prompt_embeds,  # (B, 768)
    encoder_hidden_states=prompt_embeds,      # (B, seq_len, 4096)
    txt_ids=text_ids,                  # (seq_len, 3) — positional IDs for text
    img_ids=latent_image_ids,          # (num_patches, 3) — positional IDs for image
)
```

---

## 4. LoRA adapter

Instead of fine-tuning the entire transformer (~12B parameters), a LoRA adapter is added to specific layers.

### Configuration

```python
LoraConfig(
    r=32,                    # rank
    lora_alpha=32,           # scaling (alpha / r = 1.0)
    init_lora_weights="gaussian",
    target_modules=[
        "x_embedder",          # the expanded input projection
        "attn.to_k",           # key projection
        "attn.to_q",           # query projection
        "attn.to_v",           # value projection
        "attn.to_out.0",       # output projection
        "attn.add_k_proj",     # cross-attention key
        "attn.add_q_proj",     # cross-attention query
        "attn.add_v_proj",     # cross-attention value
        "attn.to_add_out",     # cross-attention output
        "ff.net.0.proj",       # feedforward gate
        "ff.net.2",            # feedforward output
        "ff_context.net.0.proj",  # context feedforward gate
        "ff_context.net.2",       # context feedforward output
    ],
)
```

### What this means

- Every attention layer (self-attention and cross-attention) and every feedforward layer gets a low-rank update.
- The base model weights stay **frozen** during training.
- The trainable parameter count is roughly `2 * rank * (in_features + out_features)` per target module, which is a tiny fraction of the full model.
- The saved LoRA file is typically ~100-200 MB (vs ~24 GB for the full model).

---

## 5. Text encoders

Two text encoders produce two different representations:

### CLIP (clip-vit-large-patch14)

- Produces a single **pooled** embedding vector per prompt: `(B, 768)`
- Used as `pooled_projections` in the transformer
- Maps the overall semantic meaning of the prompt

### T5 (t5-v1_1-xxl)

- Produces a **sequence** of token embeddings: `(B, seq_len, 4096)`
- `seq_len` is up to 512 tokens (padded)
- Used as `encoder_hidden_states` in cross-attention
- Provides fine-grained token-level guidance

For object removal, the prompt is typically just `"There is nothing here."` — the text encoders still run but the guidance is minimal. The real conditioning comes from the image/mask channels.

---

## 6. Scheduler — FlowMatchEulerDiscreteScheduler

This is **not** a standard DDPM or DDIM scheduler. It uses flow matching:

- Noise is added as a linear interpolation: `noisy = (1 - sigma) * clean + sigma * noise`
- The model predicts the velocity field (noise - clean), not the noise directly
- Timesteps are sigmas in [0, 1], spaced using a shift function based on image resolution
- Default is **28 steps** at inference

### Sigma schedule

```python
sigmas = np.linspace(1.0, 1/num_steps, num_steps)
mu = calculate_shift(image_seq_len)  # adjusts schedule based on resolution
timesteps = scheduler.set_timesteps(sigmas=sigmas, mu=mu)
```

---

## 7. Tensor shape summary (1024x1024 input)

| Stage | Tensor | Shape |
|-------|--------|-------|
| Input image | PIL → tensor | `(1, 3, 1024, 1024)` |
| VAE encoded | image_latents | `(1, 16, 128, 128)` |
| Mask downsampled | mask_latents | `(1, 16, 128, 128)` |
| Control concat | control_image | `(1, 48, 128, 128)` |
| Packed control | packed_control | `(1, 4096, 192)` |
| Noise latents | latents | `(1, 4096, 64)` |
| Transformer input | hidden_states | `(1, 4096, 256)` |
| Text (CLIP) | pooled_prompt_embeds | `(1, 768)` |
| Text (T5) | prompt_embeds | `(1, 512, 4096)` |
| Image pos IDs | latent_image_ids | `(4096, 3)` |
| Transformer output | noise_pred | `(1, 4096, 64)` |
| Unpacked | latents | `(1, 16, 128, 128)` |
| VAE decoded | output image | `(1, 3, 1024, 1024)` |

The 4096 patches come from: `(128/2) * (128/2) = 64 * 64 = 4096` (2x2 patch packing).
