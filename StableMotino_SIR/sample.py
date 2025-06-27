
import argparse
import math
import random
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from pathlib import Path
import os
import logging
from tqdm import tqdm

from accelerate import Accelerator
import transformers
import datasets
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil
import time

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
from matplotlib.colors import hsv_to_rgb

from packaging import version
from torchvision import transforms
# from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

import sys
sys.path.append("..")
from dataloader.dataset_configuration import prepare_dataset,Disparity_Normalization,resize_max_res_tensor

from recPipeline import SDRecPipeline
from PIL import Image
import cv2

logger = get_logger(__name__, log_level="INFO")

def flow_to_image(flow, max_flow=256):
    # flow shape (H, W, C)
    if max_flow is not None:
        max_flow = max(max_flow, 1.0)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)

def sample(pretrained_model_name_or_path,vae,text_encoder,tokenizer, denoise_steps,
                   unet,accelerator,scheduler,output_dir,input_image_path,input_mask_path,weight_dtype=torch.float16,
                   partial_fp16=False,
                   normalized_input=False
                   ):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    input_image_path = Path(input_image_path)
    input_mask_path = Path(input_mask_path)
    files = [p.name for p in input_image_path.glob("*.jpg")]

    denoise_steps = int(denoise_steps)

    logger.info("Running validation ... ")

    pipeline = SDRecPipeline.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
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
        for idx in tqdm(range(len(files))):
            single_image_path = input_image_path / files[idx]
            single_mask_path = input_mask_path / files[idx]
            input_image_pil = Image.open(single_image_path)
            input_mask_pil = Image.open(single_mask_path)

            pipe_out = pipeline(input_image_pil,input_mask_pil,
                denosing_steps=denoise_steps,
                show_progress_bar = True,
                )

            img_pred = pipe_out.warped_img # warpped
            # flow_show = pipe_out.flow_np.cpu()
            # flow_show = flow_to_image(flow_show.permute(1,2,0))
            
            single_name = files[idx].rstrip(".jpg")
            img_save_path = os.path.join(
                output_dir, f"{single_name}.png"
            )
            # flow_save_path = os.path.join(
            #     output_dir, f"flow/{single_name}.png"
            # )
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            # os.makedirs(os.path.join(output_dir, 'flow'), exist_ok=True)
            if os.path.exists(img_save_path):
                logging.warning(
                    f"Existing file: '{img_save_path}' will be overwritten"
                )
            # if os.path.exists(flow_save_path):
            #     logging.warning(
            #         f"Existing file: '{flow_save_path}' will be overwritten"
            #     )
            save_image(img_pred, img_save_path)
            # cv2.imwrite(flow_save_path, flow_show)
            
            del img_pred
        del pipeline
        torch.cuda.empty_cache()

if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num",
            type=str,
            default=None,
            required=True,
        )
        args = parser.parse_args()
        return args
    args = parse_args()

    output_dir = f'./results'
    partial_fp16 = False
    normalized_input = True

    pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2'
    checkpoint = f'../Checkpoint-SIR'

    input_image_path = './DIR-D/testing/input'
    input_mask_path = './DIR-D/testing/mask'  

    weight_dtype = torch.float32
    accelerator = Accelerator()    

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path,
                                        subfolder='vae',local_files_only=True)
    accelerate.load_checkpoint_in_model(vae, os.path.join(checkpoint, 'vae'))

    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,
                                                 subfolder='text_encoder',local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(checkpoint,subfolder="unet_ema",
                                                in_channels=12, sample_size=(48, 64),
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True,local_files_only=True)
    
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path,subfolder='scheduler',timestep_spacing='trailing',local_files_only=True)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,subfolder='tokenizer',local_files_only=True)

    logger.info("loading the noise scheduler and the tokenizer from {}".format(pretrained_model_name_or_path),main_process_only=True)

    # Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.eval()


    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if partial_fp16:
        vae.to(accelerator.device, dtype=torch.float16)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    sample(pretrained_model_name_or_path=pretrained_model_name_or_path, vae=vae, text_encoder=text_encoder, 
           tokenizer=tokenizer, denoise_steps=args.num, unet=unet, accelerator=accelerator, scheduler=noise_scheduler, 
           output_dir=output_dir, input_image_path=input_image_path, input_mask_path=input_mask_path,
           weight_dtype=weight_dtype,
           partial_fp16=partial_fp16,
           normalized_input=normalized_input)
