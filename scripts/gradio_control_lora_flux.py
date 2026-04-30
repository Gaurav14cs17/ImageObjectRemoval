import io
import os
import torch
import random

import gradio as gr
import numpy as np

from PIL import Image, ImageCms
import torch
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
from omnieraser.pipeline_flux_control_removal import FluxControlRemovalPipeline

torch.set_grad_enabled(False)
os.environ['GRADIO_TEMP_DIR'] = './tmp'

image_path = mask_path = None
image_examples = [...] 
image_path = mask_path =None
image_examples = [
    [
        "example/image/3c43156c-2b44-4ebf-9c47-7707ec60b166.png",
        "example/mask/3c43156c-2b44-4ebf-9c47-7707ec60b166.png"
    ],
    [
        "example/image/0e5124d8-fe43-4b5c-819f-7212f23a6d2a.png",
        "example/mask/0e5124d8-fe43-4b5c-819f-7212f23a6d2a.png"
    ],
    [
        "example/image/0f900fe8-6eab-4f85-8121-29cac9509b94.png",
        "example/mask/0f900fe8-6eab-4f85-8121-29cac9509b94.png"
    ],
    [
        "example/image/3ed1ee18-33b0-4964-b679-0e214a0d8848.png",
        "example/mask/3ed1ee18-33b0-4964-b679-0e214a0d8848.png"
    ],
    [
        "example/image/9a3b6af9-c733-46a4-88d4-d77604194102.png",
        "example/mask/9a3b6af9-c733-46a4-88d4-d77604194102.png"
    ],
    [
        "example/image/87cdf3e2-0fa1-4d80-a228-cbb4aba3f44f.png",
        "example/mask/87cdf3e2-0fa1-4d80-a228-cbb4aba3f44f.png"
    ],
    [
        "example/image/55dd199b-d99b-47a2-a691-edfd92233a6b.png",
        "example/mask/55dd199b-d99b-47a2-a691-edfd92233a6b.png"
    ]
    
]

    
def load_model(base_model_path, lora_path):
    global pipe
    transformer = FluxTransformer2DModel.from_pretrained(base_model_path, subfolder='transformer', torch_dtype=torch.bfloat16)
    gr.Info(str(f"Model loading: {int((40 / 100) * 100)}%"))
    # enable image inputs
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
        base_model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    gr.Info(str(f"Model loading: {int((80 / 100) * 100)}%"))
    gr.Info(str(f"Inject LoRA: {lora_path}"))
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
    gr.Info(str(f"Model loading: {int((100 / 100) * 100)}%"))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def predict(
    input_image,
    prompt,
    ddim_steps,
    seed,
    scale,
    image_paths,
    mask_paths

):
    global image_path, mask_path
    gr.Info(str(f"Set seed = {seed}"))
    if image_paths is not None:
        input_image["background"] = load_image(image_paths).convert("RGB")
        input_image["layers"][0] = load_image(mask_paths).convert("RGB")
        
    size1, size2 = input_image["background"].convert("RGB").size
    icc_profile = input_image["background"].info.get('icc_profile')
    if icc_profile:
        gr.Info(str(f"Image detected to contain ICC profile, converting color space to sRGB..."))
        srgb_profile = ImageCms.createProfile("sRGB")
        io_handle = io.BytesIO(icc_profile)   
        src_profile = ImageCms.ImageCmsProfile(io_handle)  
        input_image["background"] = ImageCms.profileToProfile(input_image["background"], src_profile, srgb_profile)
        input_image["background"].info.pop('icc_profile', None)

    if size1 < size2:
        input_image["background"] = input_image["background"].convert("RGB").resize((1024, int(size2 / size1 * 1024)))
    else:
        input_image["background"] = input_image["background"].convert("RGB").resize((int(size1 / size2 * 1024), 1024))

    img = np.array(input_image["background"].convert("RGB"))

    H = int(np.shape(img)[0] - np.shape(img)[0] % 16)
    W = int(np.shape(img)[1] - np.shape(img)[1] % 16)

    input_image["background"] = input_image["background"].resize((W, H))
    input_image["layers"][0] = input_image["layers"][0].resize((W, H))

    if seed == -1:
        seed = random.randint(1, 2147483647)
        set_seed(random.randint(1, 2147483647))
    else:
        set_seed(seed)
    if image_paths is None:
        img=input_image["layers"][0]
        img_data = np.array(img)
        alpha_channel = img_data[:, :, 3]
        white_background = np.ones_like(alpha_channel) * 255
        gray_image = white_background.copy()
        gray_image[alpha_channel == 0] = 0
        gray_image_pil = Image.fromarray(gray_image).convert('L')
    else:
        gray_image_pil = input_image["layers"][0]
    result = pipe(
        prompt=prompt,
        control_image=input_image["background"].convert("RGB"),
        control_mask=gray_image_pil.convert("RGB"),
        width=W,
        height=H,
        num_inference_steps=ddim_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        guidance_scale=scale,
        max_sequence_length=512,
    ).images[0]

    mask_np = np.array(input_image["layers"][0].convert("RGB"))
    red = np.array(input_image["background"]).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(input_image["background"])
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )

    dict_res = [input_image["background"], input_image["layers"][0], result_m, result]

    dict_out = [result]
    image_path = None
    mask_path = None
    return dict_out, dict_res
   

def infer(
    input_image,
    ddim_steps,
    seed,
    scale,
    removal_prompt,

):
    img_path = image_path
    msk_path = mask_path
    return predict(input_image, 
                   removal_prompt, 
                   ddim_steps, 
                   seed,
                   scale,
                   img_path,
                   msk_path
    )

def process_example(image_paths, mask_paths):
    global image_path, mask_path
    image = Image.open(image_paths).convert("RGB")
    mask = Image.open(mask_paths).convert("L") 
    black_background = Image.new("RGB", image.size, (0, 0, 0))
    masked_image = Image.composite(black_background, image, mask)
    
    image_path = image_paths
    mask_path = mask_paths
    return masked_image
custom_css = """

.contain { max-width: 1200px !important; }

.custom-image {
    border: 2px dashed #7e22ce !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.custom-image:hover {
    border-color: #9333ea !important;
    box-shadow: 0 4px 15px rgba(158, 109, 202, 0.2) !important;
}

.btn-primary {
    background: linear-gradient(45deg, #7e22ce, #9333ea) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
}
#inline-examples {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 8px !important;
}

#inline-examples .thumbnail {
    border-radius: 8px !important;
    transition: transform 0.2s ease !important;
}

#inline-examples .thumbnail:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.example-title h3 {
    margin: 0 0 12px 0 !important;
    color: #475569 !important;
    font-size: 1.1em !important;
    display: flex !important;
    align-items: center !important;
}

.example-title h3::before {
    content: "📚";
    margin-right: 8px;
    font-size: 1.2em;
}

.row { align-items: stretch !important; }

.panel { height: 100%; }
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="purple",
        font=[gr.themes.GoogleFont('Inter'), 'sans-serif']
    ),
    title="Omnieraser"
) as demo:
    base_model_path = 'black-forest-labs/FLUX.1-dev'
    lora_path = 'theSure/Omnieraser'
    load_model(base_model_path=base_model_path, lora_path=lora_path)

    ddim_steps = gr.Slider(visible=False, value=28)
    scale = gr.Slider(visible=False, value=3.5)
    seed = gr.Slider(visible=False, value=-1)
    removal_prompt = gr.Textbox(visible=False, value="There is nothing here.")

    gr.Markdown("""
    <div align="center">
        <h1 style="font-size: 2.5em; margin-bottom: 0.5em;">🪄 Omnieraser</h1>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## 📥 Input Panel")
            
            with gr.Group():
                input_image = gr.Sketchpad(
                    sources=["upload"],
                    type="pil",
                    label="Upload & Annotate",
                    elem_id="custom-image",
                    interactive=True
                )
            with gr.Row(variant="compact"):
                run_button = gr.Button(
                    "🚀 Start Processing",
                    variant="primary",
                    size="lg"
                )
            with gr.Group():
                gr.Markdown("### ⚙️ Control Parameters")
                seed = gr.Slider(
                    label="Random Seed",
                    minimum=-1,
                    maximum=2147483647,
                    value=1234,
                    step=1,
                    info="-1 for random generation"
                )
            with gr.Column(variant="panel"):
                gr.Markdown("### 🖼️ Example Gallery", elem_classes=["example-title"])
                example = gr.Examples(
                    examples=image_examples,
                    inputs=[
                        gr.Image(label="Image", type="filepath",visible=False),
                        gr.Image(label="Mask", type="filepath",visible=False)
                    ],
                    outputs=[input_image],
                    fn=process_example,
                    run_on_click=True,
                    examples_per_page=10,
                    label="Click any example to load",
                    elem_id="inline-examples"
                )

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## 📤 Output Panel")
            with gr.Tabs():
                with gr.Tab("Final Result"):
                    inpaint_result = gr.Gallery(
                        label="Generated Image",
                        columns=2,
                        height=450,
                        preview=True,
                        object_fit="contain"
                    )

                with gr.Tab("Visualization Steps"):
                    gallery = gr.Gallery(
                        label="Workflow Steps",
                        columns=2,
                        height=450,
                        object_fit="contain"
                    )

    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            ddim_steps,
            seed,
            scale,
            removal_prompt,
        ],
        outputs=[inpaint_result, gallery]
    )
    

if __name__ == '__main__':
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7999,
        show_api=False,
    )