LAUNCH_TRAINING(){

# accelerate config: default
tracker_project_name='DIR-D-zero2-plus-fp32-perceptualLoss'
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
flow_datapath='./MDM_Flow'
condition_datapath='./DIR-D/training/input'
mask_datapath='./DIR-D/training/mask'
gt_datapath='./DIR-D/training/gt'
train_condition_list='./DIR-D/training/input_list.txt'
train_flow_list='./MDM-2step-training/flow_list.txt'
train_mask_list='./DIR-D/training/mask_list.txt'
train_gt_list='./DIR-D/training/gt_list.txt'
output_dir='../checkpoints'
train_batch_size=16
num_train_epochs=5000
gradient_accumulation_steps=1
learning_rate=1e-5
lr_warmup_steps=1000
dataloader_num_workers=4


TORCH_DISTRIBUTED_DEBUG=INFO NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 accelerate launch --config_file accelerate_configs/sdrec.yaml trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --flow_datapath $flow_datapath \
                  --condition_datapath $condition_datapath\
                  --mask_datapath $mask_datapath \
                  --gt_datapath $gt_datapath \
                  --train_flow_list $train_flow_list \
                  --train_mask_list $train_mask_list \
                  --train_condition_list $train_condition_list \
                  --train_gt_list $train_gt_list \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --checkpointing_steps 2000 \
                  --use_ema \
                  --resume_from_checkpoint "latest" \
                  --report_to "tensorboard" \
                  --enable_xformers_memory_efficient_attention \
                  --use_deepspeed \
                  --normalized_input \
                  --multi_timestep \
                  --multi_res_noise \
                  --mr_noise_strength 0.9 \
                  --annealed_mr_noise \
                  --mr_noise_downscale_strategy original
}

LAUNCH_TRAINING
