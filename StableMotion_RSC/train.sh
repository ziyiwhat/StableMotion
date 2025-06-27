LAUNCH_TRAINING(){

# accelerate config: default
tracker_project_name='022301-StableRS-RealRS-bs32-2A100-lr1e-5-3Loss'
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
datapath='./Real_RS/train'
train_batch_size=32
num_train_epochs=1000
gradient_accumulation_steps=1
learning_rate=1e-5
lr_warmup_steps=1000
dataloader_num_workers=4


TORCH_DISTRIBUTED_DEBUG=INFO NCCL_DEBUG=INFO accelerate launch trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --datapath $datapath \
                  --output_dir $tracker_project_name \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --checkpointing_steps 5000 \
                  --use_ema \
                  --resume_from_checkpoint "latest" \
                  --report_to "wandb" \
                  --use_deepspeed \
                  --normalized_input \
                  --multi_timestep \
                  --multi_res_noise \
                  --mr_noise_strength 0.9 \
                  --annealed_mr_noise \
                  --mr_noise_downscale_strategy original
                #   --enable_xformers_memory_efficient_attention \
}

LAUNCH_TRAINING
