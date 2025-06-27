import os
import argparse
import math
import random
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

import logging
import tqdm
from tqdm import tqdm as tqdm_tqdm

from accelerate import Accelerator
import transformers
import datasets
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil


import diffusers
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import inspect
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

import sys
sys.path.append("..")
from dataloader.dataset_configuration import prepare_dataset

from recPipeline import SDRecPipeline
from PIL import Image
from vgg_loss import vgg_loss
import torch.nn.init as init

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

memory_logging = False
# memory_logging = True

def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

def memory_show(prefix=""):
    if memory_logging is False:
        return
    local_rank = dist.get_rank()
    if local_rank != 0:
        return
    allocated_memory = torch.cuda.memory_allocated(local_rank) / (1024.0 ** 3)
    reserved_memory = torch.cuda.memory_reserved(local_rank) / (1024.0 ** 3)
    max_allocated_memory = torch.cuda.max_memory_allocated(local_rank) / (1024.0 ** 3)
    max_reserved_memory = torch.cuda.max_memory_reserved(local_rank) / (1024.0 ** 3)
    print(f"[rank: {local_rank}] {prefix} allocated_memory: {allocated_memory}, {max_allocated_memory} GB, reserved_memory: {reserved_memory}, {max_reserved_memory} GB")

# multi-res noise
def multi_res_noise_like(
    x, strength=0.9, downscale_strategy="original", generator=None, device=None
):
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))
    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=x.device, generator=generator).to(x)

    if "original" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "every_layer" == downscale_strategy:
        for i in range(int(math.log2(min(w, h)))):
            w, h = max(1, int(w / 2)), max(1, int(h / 2))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
    elif "power_of_two" == downscale_strategy:
        for i in range(10):
            r = 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "random_step" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r))), max(1, int(h / (r)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    else:
        raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    return noise

from typing import Optional, Tuple
from diffusers.utils import is_torch_version
def decoder_custom_forward(
    self,
    sample: torch.FloatTensor,
    latent_embeds: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    r"""The forward method of the `Decoder` class."""
    sample = self.conv_in(sample)

    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    if self.gradient_checkpointing:
    # if self.training and self.gradient_checkpointing:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        if is_torch_version(">=", "1.11.0"):
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                sample,
                latent_embeds,
                use_reentrant=False,
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
        else:
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, latent_embeds
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
    else:
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample

setattr(diffusers.models.autoencoders.vae.Decoder, "forward", decoder_custom_forward)


def initialize_weights(model):
	for m in model.modules():
		# if Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.zeros_(m.weight.data)
			# if bias
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()	

def flow_warp(x, flow12, pad="zeros", mode="bilinear"):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if "align_corners" in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()
    """ print(torch.max(v_grid))
    print(torch.min(v_grid)) """
    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    """ print(torch.max(v_grid_norm))
    print(torch.min(v_grid_norm)) """
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def log_validation(vae,text_encoder,tokenizer,unet,args,accelerator,weight_dtype,scheduler,epoch,
                   input_image_path="/vepfs-d-data/q-jigan/hmb/stablerec/StableRS/rs_validation.jpg"
                   ):
    # TODO: denoise_steps
    denoise_steps = 1
    batch_size = 1
    partial_fp16 = args.partial_fp16
    normalized_input = args.normalized_input
    
    
    logger.info("Running validation ... ")
    pipeline = SDRecPipeline.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                   vae=accelerator.unwrap_model(vae),
                                                   text_encoder=accelerator.unwrap_model(text_encoder),
                                                   tokenizer=tokenizer,
                                                   unet = accelerator.unwrap_model(unet),
                                                   safety_checker=None,
                                                   scheduler = accelerator.unwrap_model(scheduler),
                                                   weight_dtype=weight_dtype,
                                                   partial_fp16=partial_fp16,
                                                   normalized_input=normalized_input
                                                   )

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        input_image_pil = Image.open(input_image_path).convert('RGB')

        pipe_out = pipeline(input_image_pil,
             denosing_steps=denoise_steps,
             batch_size = batch_size,
             show_progress_bar = True,
             )

        # flow_pred = pipe_out.flow_np # flow
        warped_img = pipe_out.warped_img # warpped
        
        # savd as npy
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
        pred_name_base = rgb_name_base + "_pred"

        # Colorize
        colored_save_path = os.path.join(
            args.output_dir, f"{pred_name_base}_{epoch}_vaesolotrained_warpped.png"
        )
        if os.path.exists(colored_save_path):
            logging.warning(
                f"Existing file: '{colored_save_path}' will be overwritten"
            )
        save_image(warped_img, colored_save_path)
        
        del warped_img
        del pipeline
        torch.cuda.empty_cache()





def parse_args():
    parser = argparse.ArgumentParser(description="Repurposing Diffusion-Based Image Generators for Monocular flow Estimation")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--flow_datapath",
        type=str,
        required=False,
        help="The Root Dataset Path.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="The Root Dataset Path.",
    )
    parser.add_argument(
        "--mask_datapath",
        type=str,
        required=False,
        help="The mask Dataset Path.",        
    )
    parser.add_argument(
        "--gt_datapath",
        type=str,
        required=False,
        help="The gt Dataset Path.",        
    )
    parser.add_argument(
        "--condition_datapath",
        type = str,
        required=False
    )
    parser.add_argument(
        "--train_condition_list",
        type = str,
        required=False
    )
    parser.add_argument(
        "--train_flow_list",
        type=str,
        required=False
    )
    parser.add_argument(
        "--train_mask_list",
        type=str,
        required=False       
    )
    parser.add_argument(
        "--train_gt_list",
        type=str,
        required=False       
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    # using EMA for improving the generalization
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # using xformers for efficient training 
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # noise offset?::: #TODO HERE
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # validations every 5 Epochs
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    # 下方是 Lin Sui 新添加的训练参数
    # 
    parser.add_argument(
        "--use_deepspeed",
        action="store_true"
    )

    # bf16 时候使用, 在 VAE 部分使用 fp16, 其余部分使用 bf16, 但是这样训练会导致精度下降, maybe 和反传有关
    parser.add_argument(
        "--partial_fp16",
        action="store_true"
    )

    # 输入 VAE 的 input 是否要 normalized 到 -1~1 区间
    parser.add_argument(
        "--normalized_input",
        action="store_true"
    )

    # 训练时是否添加 flip data augmentation & flip 的概率
    parser.add_argument(
        "--flip_input",
        action="store_true"
    )

    parser.add_argument(
        "--flip_p",
        type=float,
        default=0.5
    )

    # 训练时 timestep 采样多个, 而不是一个
    parser.add_argument(
        "--multi_timestep",
        action="store_true"
    )

    # marigold 的 random generator, TODO: 需要想一下这东西有没有必要
    parser.add_argument(
        "--use_random_generator",
        action="store_true"
    )
    
    # 是否使用 multi-res noise
    parser.add_argument(
        "--multi_res_noise",
        action="store_true"
    )
    # multi-res noise 的 强度
    parser.add_argument(
        "--mr_noise_strength",
        type=float,
        default=0.9
    )
    # 是否使用 annealed multi-res noise
    parser.add_argument(
        "--annealed_mr_noise",
        action="store_true"
    )

    parser.add_argument(
        "--mr_noise_downscale_strategy",
        type=str,
        default="original",
        choices=["original", "every_layer", "power_of_two", "random_step"]
    )

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_path is None:
    #     raise ValueError("Need either a dataset name or a DataPath.")

    return args
    
    
def main():    
    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()

    # save  the tensorboard log files
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # tell the gradient_accumulation_steps, mix precison, and tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs
    # set the warning levels
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler',timestep_spacing='trailing',local_files_only=False)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer',local_files_only=False)
    logger.info("loading the noise scheduler and the tokenizer from {}".format(args.pretrained_model_name_or_path),main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    SD_flag = True
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # accelerator.register_for_checkpointing(vae)
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder',local_files_only=False)
        # Use from_config method to train from scratch: https://github.com/huggingface/diffusers/discussions/8458
        if SD_flag:
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                    subfolder='vae',local_files_only=False)
            unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                        in_channels=4, sample_size=(32, 32),
                                                        low_cpu_mem_usage=False,
                                                        ignore_mismatched_sizes=True,local_files_only=False)
            in_channels = 8
            out_channels = unet.conv_in.out_channels
            unet.register_to_config(in_channels=in_channels)

            with torch.no_grad():
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
                new_conv_in.weight[:, 4:, :, :].copy_(unet.conv_in.weight)
                new_conv_in.weight[:, :, :, :] = new_conv_in.weight[:, :, :, :] / 2.0

                unet.conv_in = new_conv_in
        if not SD_flag:
            vae_config = AutoencoderKL.load_config('./vae_config.json')
            vae = AutoencoderKL.from_config(vae_config)
            # unet_config = UNet2DConditionModel.load_config(args.pretrained_model_name_or_path, subfolder="unet")
            unet_config = UNet2DConditionModel.load_config('./unet_config.json')
            unet = UNet2DConditionModel.from_config(unet_config, in_channels=8, sample_size=(32, 32))

    # Freeze vae and text_encoder and set unet to trainable.
    vae.requires_grad_(False)
    # vae.train()
    text_encoder.requires_grad_(False)
    unet.train() # only make the unet-trainable
    # unet.requires_grad_(False)

    # vae.requires_grad_(False)
    # vae.encoder.requires_grad_(True)
    # vae.encoder.train()
    # text_encoder.requires_grad_(False)
    # unet.requires_grad_(False) # only make the unet-trainable
    
    if args.use_ema:
        if SD_flag:
            ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                        in_channels=4, sample_size=(32, 32),
                                                        low_cpu_mem_usage=False,
                                                        ignore_mismatched_sizes=True,local_files_only=False)
            in_channels = 8
            out_channels = ema_unet.conv_in.out_channels
            ema_unet.register_to_config(in_channels=in_channels)
            with torch.no_grad():
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels, ema_unet.conv_in.kernel_size, ema_unet.conv_in.stride, ema_unet.conv_in.padding
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(ema_unet.conv_in.weight)
                new_conv_in.weight[:, 4:8, :, :].copy_(ema_unet.conv_in.weight)
                new_conv_in.weight[:, :, :, :] = new_conv_in.weight[:, :, :, :] / 2.0

                ema_unet.conv_in = new_conv_in
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
            ema_unet.to(accelerator.device)
        if not SD_flag:
            # unet_config = UNet2DConditionModel.load_config(args.pretrained_model_name_or_path, subfolder="unet")
            unet_config = UNet2DConditionModel.load_config('./unet_config.json')
            ema_unet = UNet2DConditionModel.from_config(unet_config, in_channels=8, sample_size=(32, 32))
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
            ema_unet.to(accelerator.device)
            
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                # unet.save_pretrained(os.path.join(output_dir, "unet"))
                # vae.save_pretrained(os.path.join(output_dir, "vae"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # using checkpint  for saving the memories
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # optimizer settings
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        (train_loader, test_loader), dataset_config_dict = prepare_dataset(
            datapath = args.datapath,
            batch_size=args.train_batch_size,
            datathread=args.dataloader_num_workers,
            logger=logger,
            normalized_input=args.normalized_input,
            flip_input=args.flip_input,
            flip_p=args.flip_p
        )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_loader, test_loader,lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, test_loader,lr_scheduler
    )
    crit_vgg = vgg_loss.VGGLoss().to(accelerator.device)
    # scale factor.
    rgb_latent_scale_factor = 0.18215
    flow_latent_scale_factor = 0.18215

    def generate_seed_sequence(
        initial_seed,
        length,
        min_val=-0x8000_0000_0000_0000,
        max_val=0xFFFF_FFFF_FFFF_FFFF,
    ):
        if initial_seed is None:
            logging.warning("initial_seed is None, reproducibility is not guaranteed")
        random.seed(initial_seed)

        seed_sequence = []
        for _ in tqdm_tqdm(range(length)):
            seed = random.randint(min_val, max_val)
            seed_sequence.append(seed)
        
        return seed_sequence

    global_seed_sequence = []
    def get_next_seed():
        if 0 == len(global_seed_sequence):
            global_seed_sequence = generate_seed_sequence(
                initial_seed = args.seed,
                length=args.max_train_steps
            )
            logging.info(
                f"Global seed sequence is generated, length={len(global_seed_sequence)}"
            )
        return global_seed_sequence.pop()
    if args.use_random_generator:
        assert args.seed is not None, "When using `--use_random_generator` flag, please set the --seed parameter"

    rand_num_generator = None
    if args.use_random_generator:
        local_seed = get_next_seed()
        rand_num_generator = torch.generator(device=accelerator.device)
        rand_num_generator.manual_seed(local_seed)
    else:
        rand_num_generator = None

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    partial_fp16 = args.partial_fp16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if partial_fp16:
        vae.to(accelerator.device, dtype=torch.float16)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        unet.eval()
        log_validation(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            scheduler=noise_scheduler,
            epoch=-1)
    
    
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() 
        train_loss = 0.0
        train_loss_latentNoise = 0.0
        train_loss_pl = 0.0
        train_loss_vgg = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                cond = batch['cond']
                flow = batch['flow']
                flow_gt = flow * 50.0
                flow_gt = flow_gt[:, :2, :, :]
                gt = batch['gt']
                
                if partial_fp16:
                    h_rgb = vae.encoder(cond.half())
                    moments_rgb = vae.quant_conv(h_rgb)
                    mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                    rgb_latents = mean_rgb *rgb_latent_scale_factor
                    rgb_latents = rgb_latents.to(weight_dtype)
                else:
                    h_rgb = vae.encoder(cond.to(weight_dtype))
                    moments_rgb = vae.quant_conv(h_rgb)
                    mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                    rgb_latents = mean_rgb *rgb_latent_scale_factor

                # encode disparity to lantents
                if partial_fp16:
                    h_disp = vae.encoder(flow.half())
                    moments_disp = vae.quant_conv(h_disp)
                    mean_disp, logvar_disp = torch.chunk(moments_disp, 2, dim=1)
                    disp_latents = mean_disp * flow_latent_scale_factor
                    disp_latents = disp_latents.to(weight_dtype)
                else:
                    h_disp = vae.encoder(flow.to(weight_dtype))
                    moments_disp = vae.quant_conv(h_disp)
                    mean_disp, logvar_disp = torch.chunk(moments_disp, 2, dim=1)
                    disp_latents = mean_disp * flow_latent_scale_factor
                
                memory_show("last_vae")

                # here is the setting batch size, in our settings, it can be 1.0
                bsz = disp_latents.shape[0]

                if args.multi_timestep:
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz, ),
                        device=disp_latents.device,
                        generator=rand_num_generator
                    ).long()
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=disp_latents.device)
                    timesteps = timesteps.repeat(bsz)
                    timesteps = timesteps.long()

                # if args.multi_timestep:
                #     timesteps = torch.randint(
                #         noise_scheduler.config.num_train_timesteps-1,
                #         noise_scheduler.config.num_train_timesteps,
                #         (bsz, ),
                #         device=disp_latents.device,
                #         generator=rand_num_generator
                #     ).long()
                # else:
                #     timesteps = torch.randint(noise_scheduler.config.num_train_timesteps-1, noise_scheduler.config.num_train_timesteps, (1,), device=disp_latents.device)
                #     timesteps = timesteps.repeat(bsz)
                #     timesteps = timesteps.long()
                
                if args.multi_res_noise:
                    mr_noise_strength = args.mr_noise_strength
                    annealed_mr_noise = args.annealed_mr_noise
                    mr_noise_downscale_strategy = args.mr_noise_downscale_strategy

                    strength = mr_noise_strength
                    if annealed_mr_noise:
                        strength = strength * (timesteps / noise_scheduler.config.num_train_timesteps)
                    noise = multi_res_noise_like(
                        disp_latents,
                        strength=strength,
                        downscale_strategy=mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=disp_latents.device
                    )

                else:
                    noise = torch.randn(
                        disp_latents.shape,
                        device=disp_latents.device,
                        generator=rand_num_generator
                    ).to(disp_latents)

                # add noise to the flow lantents
                noisy_disp_latents = noise_scheduler.add_noise(disp_latents, noise, timesteps)
                
                # Encode text embedding for empty prompt
                prompt = ""
                text_inputs =tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device) #[1,2]
                empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)

                memory_show("text_encode")
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(disp_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                batch_empty_text_embed = empty_text_embed.repeat((noisy_disp_latents.shape[0], 1, 1))  # [B, 2, 1024]
                
                # predict the noise residual and compute the loss.
                unet_input = torch.cat([rgb_latents, noisy_disp_latents], dim=1)  # this order is important: [1,8,H,W]
                
                # predict the noise residual
                noise_pred = unet(unet_input, 
                                  timesteps, 
                                  encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]
                
                memory_show("after unet")
                loss_latentNoise = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                noise_scheduler.set_timesteps(50)
                flow_latents = []
                for ind in range(bsz):
                    flow_latent_local = noise_scheduler.step(noise_pred[ind].unsqueeze(0), timesteps[ind], noisy_disp_latents[ind].unsqueeze(0)).pred_original_sample
                    flow_latents.append(flow_latent_local)
                flow_latent = torch.cat(flow_latents, dim=0)

                flow_latent = flow_latent / flow_latent_scale_factor
                if partial_fp16:
                    flow_latent = flow_latent.half()
                else:
                    flow_latent = flow_latent.to(weight_dtype)

                z = vae.post_quant_conv(flow_latent)
                flow_vae_out = vae.decoder(z)
                memory_show("after decode")

                flow_pred = flow_vae_out * 50 # vital factor
                flow_pred = flow_pred[:, :2, :, :]

                pred_sample = flow_warp(cond, flow_pred)
                ones_mask = torch.ones_like(cond)
                ones_mask = ones_mask[:, 0:1, :, :]
                ones_mask = flow_warp(ones_mask, flow_gt)
                pred_sample = ones_mask * pred_sample

                loss_pl = F.mse_loss(gt, pred_sample, reduction='mean')
                loss_vgg = crit_vgg(pred_sample, gt, target_is_features=False)

                if args.normalized_input:
                    loss = loss_latentNoise + loss_pl + loss_vgg * 0.01
                else:
                    loss = loss_latentNoise + loss_pl + loss_vgg * 0.04
                # loss = loss_latentNoise + loss_pl
                # loss = loss_latentNoise
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_loss_latentNoise = accelerator.gather(loss_latentNoise.repeat(args.train_batch_size)).mean()
                train_loss_latentNoise = avg_loss_latentNoise.item() / args.gradient_accumulation_steps
                avg_loss_pl = accelerator.gather(loss_pl.repeat(args.train_batch_size)).mean()
                train_loss_pl = avg_loss_pl.item() / args.gradient_accumulation_steps
                avg_loss_vgg = accelerator.gather(loss_vgg.repeat(args.train_batch_size)).mean()
                train_loss_vgg = avg_loss_vgg.item() / args.gradient_accumulation_steps

                # Backpropagate
                memory_show("before loss backward")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                memory_show("after optimizer")


            # currently the EMA is not used.
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"train_loss_latentNoise": train_loss_latentNoise}, step=global_step)
                accelerator.log({"train_loss_pl": train_loss_pl}, step=global_step)
                accelerator.log({"train_loss_vgg": train_loss_vgg}, step=global_step)
                
                train_loss = 0.0
                train_loss_latentNoise = 0.0
                train_loss_pl = 0.0
                train_loss_vgg = 0.0

                # saving the checkpoints
                if args.use_deepspeed or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        print("Saving checkpoint...")
                # if args.use_deepspeed or global_step % args.checkpointing_steps == 0:
                    # if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # accelerator.save_model(unet, '../models/unet')
                        # accelerator.save_model(vae, os.path.join(save_path, 'vae'))
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            # validation each epoch by calculate the epe and the visualization flow
            if args.use_ema:    
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            
            if epoch % 1 == 0:
                # validation inference here
                log_validation(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    scheduler=noise_scheduler,
                    epoch=epoch,
                )
            
            
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())


    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
if __name__=="__main__":
    main()
