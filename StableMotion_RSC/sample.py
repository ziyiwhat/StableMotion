
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
from torchvision import transforms as T
import diffusers
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

import inspect

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

def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u = u * u_scale
        v = v * v_scale
        res = torch.cat([u, v], dim=1)
    return res

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
                   unet,accelerator,scheduler,output_dir,input_image_path,weight_dtype=torch.float16,
                   partial_fp16=False,
                   normalized_input=False
                   ):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    input_image_path = Path(input_image_path)
    files = [p.name for p in input_image_path.glob("*.npz")]

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

    trans = T.ToTensor()
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for idx in tqdm(range(len(files))):
            single_image_path = input_image_path / files[idx]
            jpg_path = str(single_image_path).replace("npz", "jpg")
            jpg = Image.open(jpg_path)
            jpg_tensor = trans(jpg).unsqueeze(0).to(accelerator.device)
            print(files[idx])
            input_image = Image.fromarray(np.load(single_image_path)['condition'])

            pipe_out = pipeline(input_image,
                denosing_steps=denoise_steps,
                show_progress_bar = True
                )

            img_pred = pipe_out.warped_img # warpped
            flow_pred = pipe_out.flow_np.unsqueeze(0)

            flow_resized = upsample2d_flow_as(flow_pred, jpg_tensor, mode="bilinear", if_rate=True).to(accelerator.device)
            jpg_warped = flow_warp(jpg_tensor, flow_resized, pad="zeros", mode="bilinear").squeeze(0)
            # mask_pred = pipe_out.warped_mask
            # flow_show = pipe_out.flow_np.cpu()
            # flow_show = flow_to_image(flow_show.permute(1,2,0))
            
            single_name = files[idx].rstrip(".npz")
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

            # save_image(img_pred, img_save_path)
            save_image(jpg_warped, img_save_path)
            # save_image(mask_pred, img_save_path)
            # cv2.imwrite(flow_save_path, flow_show)
            
            del img_pred
        del pipeline
        torch.cuda.empty_cache()

if __name__ == "__main__":

    pretrained_model_name_or_path = '/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79'
    # input_image_path = '/vepfs-d-data/q-jigan/hmb/stablerec/Real_RS/test/npz' 
    input_image_path = '/root/real-rs/Real_RS/datasets--Yzl-code--RS-Diffusion/test/npz'

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--project_dir",
            type=str,
            default=None,
            required=True,
        )
        parser.add_argument(
            "--num",
            type=int,
            default=None,
            required=True,
        )
        args = parser.parse_args()
        return args
    args = parse_args()

    partial_fp16 = False
    normalized_input = True

    # dirs = os.listdir(args.project_dir)
    # dirs = [d for d in dirs if d.startswith("checkpoint")]
    # dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    # path = dirs[-1] if len(dirs) > 0 else None

    path = "checkpoint-60000"
    output_dir = f'./results/{args.project_dir}/{path}-{args.num}'
    # output_dir = f'./results/gene/reshape'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weight_dtype = torch.float32
    accelerator = Accelerator()    

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path,
                                        subfolder='vae',local_files_only=True)
    # accelerate.load_checkpoint_in_model(vae, os.path.join(checkpoint, 'vae'))

    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,
                                                 subfolder='text_encoder',local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(path,subfolder="unet_ema",
                                                in_channels=8, sample_size=(32, 32),
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
           output_dir=output_dir, input_image_path=input_image_path,
           weight_dtype=weight_dtype,
           partial_fp16=partial_fp16,
           normalized_input=normalized_input)
