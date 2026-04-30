accelerate launch --config_file accelerate.yaml train_controlnet_lora_flux.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --controlnet_model_name_or_path alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta \
    --output_dir /aaaidata/weirunpu/diffusers-0.33.0.dev0/flux_control_lora_RORD \
    --train_data_dir /huggingface/dataset_hub/RORD/train_datasets \
    --resolution 1024 \
    --guidance_scale 3.5 \
    --learning_rate 3e-5 \
    --train_batch_size 1 \
    --num_train_epochs 2 \
    --rank 32 \
    --gaussian_init_lora \
    --logging_dir tensorboard \
    --tracker_project_name flux_train_control_lora \
    --report_to tensorboard \
    --validation_steps 50 \
    --checkpointing_steps 100 \
    --use_8bit_adam \
    --allow_tf32 \
    --scale_lr \
    --lr_scheduler cosine_with_restarts \
    --mixed_precision "bf16" \
    # --resume_from_checkpoint latest \
    # --train_data_dir /huggingface/dataset_hub/VideoRemoval/train_dataset \
    # --pretrained_lora_path ckpt/LoRA-RORD/pytorch_lora_weights.safetensors \
    # --offload \
    # --brushnet_model_name_or_path /aaaidata/weirunpu/brushnet_sdxl_videoremoval/checkpoint-3000/brushnet \
