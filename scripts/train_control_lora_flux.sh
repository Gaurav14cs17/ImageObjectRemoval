accelerate launch --config_file configs/accelerate.yaml scripts/train_control_lora_flux.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --output_dir /aaaidata/weirunpu/diffusers-0.33.0.dev0/flux_control_lora_VDOR_masked \
    --train_data_dir /huggingface/dataset_hub/VideoRemoval/refine_train_dataset \
    --resolution 1024 \
    --guidance_scale 3.5 \
    --learning_rate 3e-5 \
    --train_batch_size 1 \
    --num_train_epochs 20 \
    --rank 32 \
    --gaussian_init_lora \
    --logging_dir tensorboard \
    --tracker_project_name flux_train_control_lora \
    --report_to tensorboard \
    --validation_steps 100 \
    --checkpointing_steps 100 \
    --use_8bit_adam \
    --allow_tf32 \
    --scale_lr \
    --lr_scheduler cosine_with_restarts \
    --mixed_precision "bf16" \
    --gradient_accumulation_steps 8 \
    # --pretrained_lora_path /aaaidata/weirunpu/diffusers-0.33.0.dev0/flux_control_lora_RORD/pretrained_model/pytorch_lora_weights.safetensors \
    # --resume_from_checkpoint latest \
    # --pretrained_lora_path ckpt/LoRA-RORD/pytorch_lora_weights.safetensors \
    # --train_data_dir /huggingface/dataset_hub/VideoRemoval/train_dataset \
    # --offload \

