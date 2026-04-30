# Training

Technical guide to training the Control-LoRA adapter for object removal. Covers dataset format, the training loop internals, loss function, hyperparameters, and how to launch a run.

---

## 1. Dataset format

The training script expects a folder with three sub-directories:

```
your_dataset/
├── img/          ← source images (with the object present)
├── gt_new/       ← ground truth background (same scene, object removed)
└── sam_mask/     ← binary masks (white = object region, black = background)
```

All three folders must have **matching filenames**. For example:

```
img/00001.png          ← photo with a person standing
gt_new/00001.png       ← same photo, person removed (clean background)
sam_mask/00001.png     ← white silhouette where the person was
```

The mask must be single-channel (grayscale). White (255) marks the object to remove. Black (0) marks the region to keep.

### How the training data is used

The dataset class (`TrainRemovalDataset` in `scripts/train_control_lora_flux.py`) does:

```python
mask       = Image.open("sam_mask/00001.png").convert('L')
background = Image.open("gt_new/00001.png").convert('RGB')
image      = Image.open("img/00001.png").convert('RGB')
```

- `image` = the scene with the object (input)
- `background` = the scene without the object (ground truth / target)
- `mask` = binary mask of the object

---

## 2. Data augmentation

Before encoding, the data goes through `PairedRandomCrop` (from `omnieraser/utils.py`):

```python
crop_transform = PairedRandomCrop(size=1024)
image, background, mask = crop_transform(image, background, mask)
```

This randomly crops all three images at the **same** position so they stay aligned. If the source images are smaller than 1024, they get upscaled first (with 10% margin) using bilinear interpolation for images and nearest-neighbor for masks.

After cropping, standard transforms are applied:

```python
# For images (image, background):
Resize(1024) → ToTensor() → Normalize(mean=0.5, std=0.5)
# Result: tensor in [-1, 1]

# For masks:
Resize(1024, NEAREST) → ToTensor()
# Result: tensor in [0, 1], where 1 = object region
```

Then the code constructs:

```python
masked_image = image.clone()
masked_image[mask > 0.5] = -1             # object region set to -1

foreground_image = image.clone()
foreground_image[mask < 0.5] = -1          # background set to -1

mask = 1 - mask                            # invert: 0 = inpaint, 1 = keep
```

---

## 3. What the model sees during training

Each training sample produces five tensors (all go through VAE encoding to latent space):

| Tensor | Source | Shape (latent) | Purpose |
|--------|--------|----------------|---------|
| `pixel_latents` | background (gt) | `(B, 16, 128, 128)` | The target — what the model should reconstruct |
| `pixel_masked_image_latents` | masked image | `(B, 16, 128, 128)` | Conditioning: image with object region blanked |
| `foreground_images_latents` | foreground | `(B, 16, 128, 128)` | Conditioning: just the object + its neighborhood |
| `masks` | mask | `(B, 16, 128, 128)` | Conditioning: which region to inpaint |
| `noise` | random | `(B, 16, 128, 128)` | Sampled Gaussian noise |

These get concatenated into a single input:

```python
noisy_model_input = (1 - sigma) * pixel_latents + sigma * noise

concatenated = cat([noisy_model_input,           # 16 ch (noise mixed with clean)
                    pixel_masked_image_latents,   # 16 ch
                    foreground_images_latents,     # 16 ch
                    masks],                        # 16 ch
                   dim=1)                          # total: 64 ch
```

Then packed into patches and fed to the transformer:

```python
packed = FluxControlRemovalPipeline._pack_latents(concatenated)
# shape: (B, 4096, 256) for 1024x1024 images
```

---

## 4. Loss function

The loss is **flow matching MSE**:

```python
# The target velocity is: noise - clean
target = noise - pixel_latents

# The model predicts this velocity
model_pred = transformer(packed_input, timestep, text_embeds, ...)

# Unpack prediction back to spatial layout
model_pred = FluxControlRemovalPipeline._unpack_latents(model_pred)

# Weighted MSE loss
weighting = compute_loss_weighting_for_sd3(scheme=args.weighting_scheme, sigmas=sigmas)
loss = mean(weighting * (model_pred - target) ** 2)
```

The default weighting scheme is `"none"` (uniform weighting across all timesteps). Other options: `sigma_sqrt`, `logit_normal`, `mode`, `cosmap`.

---

## 5. What gets trained (and what stays frozen)

```
FROZEN (no gradients):
  ├── VAE encoder/decoder
  ├── CLIP text encoder
  ├── T5 text encoder
  └── FLUX transformer base weights

TRAINABLE:
  └── LoRA adapter layers (injected into transformer)
      ├── x_embedder (the expanded input projection)
      ├── attn.to_k, attn.to_q, attn.to_v, attn.to_out.0
      ├── attn.add_k_proj, attn.add_q_proj, attn.add_v_proj, attn.to_add_out
      ├── ff.net.0.proj, ff.net.2
      └── ff_context.net.0.proj, ff_context.net.2
```

Optional: if `--train_norm_layers` is set, the norm scales (`norm_q`, `norm_k`, `norm_added_q`, `norm_added_k`) are also trainable.

---

## 6. Text conditioning during training

The text prompt is always `"There is nothing here."` for every training sample:

```python
captions = "There is nothing here."
prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(captions)
```

This is intentional — the model learns to remove objects based on the **image/mask conditioning**, not the text. The text is a fixed signal.

If `--proportion_empty_prompts` is set (e.g. 0.1), that fraction of batches will have the text embeddings zeroed out. This teaches the model to work with or without text guidance.

---

## 7. Hyperparameters reference

These are the defaults used in the provided shell script (`scripts/train_control_lora_flux.sh`):

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `--pretrained_model_name_or_path` | `black-forest-labs/FLUX.1-dev` | Base model to fine-tune |
| `--resolution` | `1024` | Image size (all images resized to this) |
| `--train_batch_size` | `1` | Batch size per GPU |
| `--gradient_accumulation_steps` | `8` | Effective batch size = 1 * 8 = 8 |
| `--num_train_epochs` | `20` | Total training epochs |
| `--learning_rate` | `3e-5` | Peak learning rate |
| `--scale_lr` | `true` | Scale LR by batch_size * grad_accum * num_gpus |
| `--lr_scheduler` | `cosine_with_restarts` | LR schedule shape |
| `--rank` | `32` | LoRA rank (higher = more capacity) |
| `--gaussian_init_lora` | `true` | Gaussian initialization for LoRA weights |
| `--mixed_precision` | `bf16` | Use bfloat16 for training |
| `--use_8bit_adam` | `true` | 8-bit Adam optimizer (saves ~50% optimizer memory) |
| `--allow_tf32` | `true` | Enable TF32 on Ampere GPUs |
| `--guidance_scale` | `3.5` | Guidance scale for the transformer |
| `--checkpointing_steps` | `100` | Save checkpoint every N steps |
| `--validation_steps` | `100` | Run validation every N steps |

---

## 8. How to launch training

### Minimal command

```bash
accelerate launch --config_file configs/accelerate.yaml scripts/train_control_lora_flux.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --train_data_dir /path/to/your/dataset \
    --output_dir /path/to/save/checkpoints \
    --resolution 1024 \
    --train_batch_size 1 \
    --num_train_epochs 20 \
    --rank 32 \
    --learning_rate 3e-5 \
    --mixed_precision bf16
```

### Using the provided shell script

```bash
# 1. Edit paths in the script:
vim scripts/train_control_lora_flux.sh

# 2. Change these lines:
#    --output_dir /your/output/path
#    --train_data_dir /your/dataset/path

# 3. Check GPU config:
cat configs/accelerate.yaml
# Make sure gpu_ids and num_processes match your setup

# 4. Run:
bash scripts/train_control_lora_flux.sh
```

### Resume from checkpoint

```bash
# Add to your command:
--resume_from_checkpoint latest
# or a specific checkpoint:
--resume_from_checkpoint /path/to/output_dir/checkpoint-500
```

### Load a pre-trained LoRA as starting point

```bash
--pretrained_lora_path /path/to/pytorch_lora_weights.safetensors
```

---

## 9. Accelerate config

The file `configs/accelerate.yaml` controls distributed training:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'MULTI_GPU'      # 'NO' for single GPU
gpu_ids: '0,1,2,3'                 # which GPUs to use
num_processes: 1                   # number of parallel processes
mixed_precision: 'bf16'            # matches --mixed_precision
```

For **single GPU** training, set:

```yaml
distributed_type: 'NO'
gpu_ids: '0'
num_processes: 1
```

---

## 10. Training loop internals (simplified)

```
for each epoch:
    for each batch:
        ┌─────────────────────────────────────────────────┐
        │ 1. VAE encode background → pixel_latents        │
        │ 2. VAE encode masked_image, foreground           │
        │ 3. Downsample mask to latent size                │
        │ 4. Sample random noise                           │
        │ 5. Sample random timestep t                      │
        │ 6. Mix: noisy = (1-sigma)*clean + sigma*noise    │
        │ 7. Concat: [noisy, masked, foreground, mask]     │
        │ 8. Pack into patches                             │
        │ 9. Encode text prompt                            │
        │10. Forward pass through transformer              │
        │11. Unpack prediction                             │
        │12. Loss = MSE(prediction, noise - clean)         │
        │13. Backward pass (only updates LoRA weights)     │
        │14. Optimizer step                                │
        └─────────────────────────────────────────────────┘
        
        if step % checkpointing_steps == 0:
            save checkpoint
        
        if step % validation_steps == 0:
            run validation (generate images and log to tensorboard)
```

---

## 11. Validation

During training, the script periodically generates sample outputs to check quality.

You can customize validation with:

```bash
--validation_image path/to/test_image.jpg
--validation_mask path/to/test_mask.jpg
--validation_prompt "There is nothing here."
--num_validation_images 4
--validation_steps 100
```

Validation images are logged to TensorBoard (or Weights & Biases if `--report_to wandb`).

```bash
# View training logs:
tensorboard --logdir /path/to/output_dir/tensorboard
```

---

## 12. Output

After training finishes, the output directory contains:

```
output_dir/
├── pytorch_lora_weights.safetensors    ← the trained LoRA adapter
├── checkpoint-100/                      ← intermediate checkpoint
├── checkpoint-200/
├── ...
└── tensorboard/                         ← training logs
```

To use the trained weights for inference:

```python
pipe.load_lora_weights('/path/to/output_dir', weight_name="pytorch_lora_weights.safetensors")
```

---

## 13. Memory requirements for training

| Component | VRAM (approx) |
|-----------|---------------|
| FLUX transformer (bf16) | ~12 GB |
| LoRA trainable params + gradients + optimizer states | ~4 GB (with 8-bit Adam) |
| VAE (float32) | ~160 MB |
| Text encoders (loaded on demand) | ~8 GB during text encoding |
| Activations (batch_size=1) | ~4-6 GB |
| **Total** | **~28-30 GB** (single GPU, batch_size=1) |

Tips to reduce memory:

- Use `--gradient_checkpointing` (trades compute for memory)
- Use `--offload` (moves VAE and text encoders to CPU when not in use)
- Reduce `--resolution` (e.g. 512 instead of 1024)
- Use `--use_8bit_adam` (already on by default)
