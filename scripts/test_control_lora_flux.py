import torch
from diffusers.utils import load_image, check_min_version
from diffusers import FluxTransformer2DModel
from omnieraser.pipeline_flux_control_removal import FluxControlRemovalPipeline

check_min_version("0.30.2")

# Set image path , mask path and prompt
image_path='example/image/0bce8e90-10f1-442e-8330-2917fc7fa486.png'
mask_path='example/mask/0bce8e90-10f1-442e-8330-2917fc7fa486.png'
prompt='There is nothing here.'

# Build pipeline
transformer = FluxTransformer2DModel.from_pretrained('black-forest-labs/FLUX.1-dev', 
                                                     subfolder="transformer",
                                                     torch_dtype=torch.bfloat16)

with torch.no_grad():
    initial_input_channels = transformer.config.in_channels
    new_linear = torch.nn.Linear(
        transformer.x_embedder.in_features*4,
        transformer.x_embedder.out_features,
        bias=transformer.x_embedder.bias is not None,
        dtype=transformer.dtype,
        device=transformer.device,
    )
    new_linear.weight.zero_()
    new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
    if transformer.x_embedder.bias is not None:
        new_linear.bias.copy_(transformer.x_embedder.bias)
    transformer.x_embedder = new_linear
    transformer.register_to_config(in_channels=initial_input_channels*4)

pipe = FluxControlRemovalPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
assert (
    pipe.transformer.config.in_channels == initial_input_channels*4
), f"{pipe.transformer.config.in_channels=}"

pipe.load_lora_weights('theSure/Omnieraser', 
                       weight_name="pytorch_lora_weights.safetensors",
                       )


# Load image and mask
size = (1024, 1024)
image = load_image(image_path).convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)

# Inpaint
result = pipe(
    prompt=prompt,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=generator,
    max_sequence_length=512,
    height=size[1],
    width=size[0],
).images[0]

result.save('flux_inpaint.png')
print("Successfully inpaint image")
