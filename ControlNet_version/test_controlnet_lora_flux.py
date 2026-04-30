import torch
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

name = 'I-211217_O03016_W18_F0167.jpg'
images_path = f'example/images/0bce8e90-10f1-442e-8330-2917fc7fa486.png'
masks_path = f'example/masks/0bce8e90-10f1-442e-8330-2917fc7fa486.png'
outputs_path = 'example/output/0bce8e90-10f1-442e-8330-2917fc7fa486.png'
os.makedirs(outputs_path, exist_ok=True)

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights('theSure/Omnieraser_Controlnet_version', weight_name="controlnet_flux_pytorch_lora_weights.safetensors")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)


image_path = os.path.join(images_path, name)
mask_path = os.path.join(masks_path, name)
output_path = os.path.join(outputs_path, name)
prompt='There is nothing here.'
# Load image and mask
size = (1024, 1024)
image = load_image(image_path).convert("RGB").resize(size)
#mask = load_image(mask_path).convert("RGB").resize(size)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = Image.fromarray(mask.astype(np.uint8).repeat(3,-1)).convert("RGB")
mask = mask.resize((1024, 1024))
generator = torch.Generator(device="cuda").manual_seed(24)

# Inpaint
result = pipe(
    prompt=prompt,
    height=size[1],
    width=size[0],
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    true_guidance_scale=1.0,
    guidance_scale=3.5,
    generator=generator,
    controlnet_conditioning_scale=0.9
).images[0]
result.save(output_path)

