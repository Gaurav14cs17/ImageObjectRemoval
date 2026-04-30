import io
import os
import torch
import random

import gradio as gr
import numpy as np

from PIL import Image, ImageCms
import torch
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline


torch.set_grad_enabled(False)
os.environ['GRADIO_TEMP_DIR'] = './tmp'


def load_model(base_model_path, controlnet_path, lora_path):
    global pipe
    controlnet = FluxControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
    gr.Info(str(f"Model loading: {int((20 / 100) * 100)}%"))
    transformer = FluxTransformer2DModel.from_pretrained(base_model_path, subfolder='transformer', torch_dtype=torch.bfloat16)
    gr.Info(str(f"Model loading: {int((40 / 100) * 100)}%"))
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    gr.Info(str(f"Model loading: {int((80 / 100) * 100)}%"))
    
    gr.Info(str(f"Inject LoRA: {lora_path}"))
    pipe.load_lora_weights(lora_path, weight_name="controlnet_flux_pytorch_lora_weights.safetensors")
    
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
):
    size1, size2 = input_image["image"].convert("RGB").size
    
    icc_profile = input_image["image"].info.get('icc_profile')
    if icc_profile:
        gr.Info(str(f"Image detected to contain ICC profile, converting color space to sRGB..."))
        srgb_profile = ImageCms.createProfile("sRGB")
        io_handle = io.BytesIO(icc_profile)   
        src_profile = ImageCms.ImageCmsProfile(io_handle)  
        input_image["image"] = ImageCms.profileToProfile(input_image["image"], src_profile, srgb_profile)
        input_image["image"].info.pop('icc_profile', None)

    if size1 < size2:
        input_image["image"] = input_image["image"].convert("RGB").resize((1024, int(size2 / size1 * 1024)))
    else:
        input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 1024), 1024))

    img = np.array(input_image["image"].convert("RGB"))

    H = int(np.shape(img)[0] - np.shape(img)[0] % 16)
    W = int(np.shape(img)[1] - np.shape(img)[1] % 16)

    input_image["image"] = input_image["image"].resize((W, H))
    input_image["mask"] = input_image["mask"].resize((W, H))

    if seed == -1:
        seed = random.randint(1, 2147483647)
        set_seed(random.randint(1, 2147483647))
    else:
        set_seed(seed)
    
    gr.Info(str(f"Set seed = {seed}"))

    result = pipe(
        prompt=prompt,
        control_image=input_image["image"].convert("RGB"),
        control_mask=input_image["mask"].convert("RGB"),
        width=W,
        height=H,
        num_inference_steps=ddim_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=scale # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    mask_np = np.array(input_image["mask"].convert("RGB"))
    Info = ""
    # ===============红色Mask覆盖到原始图像================================
    red = np.array(input_image["image"]).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(input_image["image"])
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )

    dict_res = [input_image["image"], input_image["mask"], result_m, result]

    gr.Info(Info)
    dict_out = [result]

    return dict_out, dict_res


def infer(
    input_image,
    ddim_steps,
    seed,
    scale,
    removal_prompt,
):
    return predict(input_image, 
                   removal_prompt, 
                   ddim_steps, 
                   seed,
                   scale)


with gr.Blocks(css="style.css") as demo:
    base_model_path = 'black-forest-labs/FLUX.1-dev'
    controlnet_model_path = 'alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta'
    lora_path = 'theSure/Omnieraser_Controlnet_version'
    load_model(base_model_path=base_model_path, 
               controlnet_path=controlnet_model_path,
               lora_path=lora_path)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input image and draw mask")
            input_image = gr.Image(source="upload", tool="sketch", type="pil")

            task = "object-removal"
            _ = gr.Text(label="Tips", value='This page is for testing purposes only.', interactive=False)

            run_button = gr.Button(label="Run object erase")
            with gr.Accordion("Advanced options", open=False):
                base_model = gr.Text(label="Base Model", value=base_model_path, interactive=True)
                controlnet_model = gr.Text(label="ControlNet Model", value=controlnet_model_path, interactive=True)

                lora_model = gr.Text(label="LoRA Model", value=lora_path, interactive=True)

                load_btn = gr.Button(value="Load the modal")
                load_btn.click(load_model, inputs=[base_model, controlnet_model, lora_model])

                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=28, step=1, interactive=True)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=45.0, value=1.0, step=0.1, interactive=True)
                seed = gr.Slider(
                    label="Seed",
                    value=1234,
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    randomize=False,
                    interactive=True
                )

                removal_prompt = gr.Textbox(label="Prompt", 
                                            value="There is nothing here.", 
                                            interactive=False,)

        with gr.Column():
            gr.Markdown("### Final Result")
            inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=2, height='auto')
            gr.Markdown("### More Visualization")
            gallery = gr.Gallery(label="Generated masks", show_label=False, columns=2, height='auto')

    
    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            ddim_steps,
            seed,
            scale,
            removal_prompt,
        ],
        outputs=[inpaint_result, gallery],
    )


if __name__ == '__main__':
    demo.queue()
    demo.launch(share=False, 
                server_name="0.0.0.0", 
                server_port=7999, 
                # debug=True,
                )
