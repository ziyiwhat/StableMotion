
from typing import Any, Dict, Union
from torchvision import transforms as T
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

import inspect

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb


def load_flow(path):
    if path.endswith(".png"):
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = flo_file[:, :, 0] == 0  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
    else:
        with open(path, "rb") as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


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

class SDRecPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular flow prediction pipeline.

    Args:
        flow_np (`np.ndarray`):
            Predicted flow map, with flow values in the range of [0, 1].
        flow_colored (`PIL.Image.Image`):
            Colorized flow map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    flow_np: np.ndarray
    warped_img: Image.Image


class SDRecPipeline(DiffusionPipeline):
    latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 weight_dtype=torch.float16,
                 partial_fp16=False,
                 normalized_input=False,
                 ):
        super().__init__()
        self.weight_dtype = weight_dtype
        self.partial_fp16 = partial_fp16
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
        self.transform = T.ToTensor()

        self.normalize_input = normalized_input
        
    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 denosing_steps: int =10,
                 batch_size:int =0,
                 show_progress_bar:bool = True,
                 ) -> SDRecPipelineOutput:
        # inherit from thea Diffusion Pipeline
        device = self.device
        
        assert denosing_steps >=1
        
        # --------------- Image Processing ------------------------
        image = self.transform(input_image)
    
        # rgb_norm = image.to(self.weight_dtype)
        rgb_norm = image
        rgb_norm = rgb_norm.to(device)
        if self.partial_fp16:
            rgb_norm = rgb_norm.half()
        else:
            rgb_norm = rgb_norm.to(self.weight_dtype)
        #rgb_norm = rgb_norm.half()

        if self.normalize_input:
            assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0
        else:
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        duplicated_rgb = torch.stack([rgb_norm])
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        
        # predicted the flow
        flow_pred_ls = []
        
        for rgb in single_rgb_loader:
            (batched_rgb,) = rgb  # here the image is still around 0-1
            flow_pred_raw = self.single_infer(
                input_rgb=batched_rgb,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
            )
            flow_pred_ls.append(flow_pred_raw.detach().clone())
        
        flow_preds = torch.concat(flow_pred_ls, axis=0).squeeze()

        torch.cuda.empty_cache()  # clear vram cache for ensembling

        flow_pred = flow_preds

        flow_pred = flow_pred * 50 # vital factor
        flow_pred = flow_pred[:2, :, :]

        warped_img = flow_warp(rgb_norm.unsqueeze(0), flow_pred.unsqueeze(0), pad="zeros", mode="bilinear").squeeze(0)
        ones = torch.ones_like(rgb_norm)
        warped_mask = flow_warp(ones.unsqueeze(0), flow_pred.unsqueeze(0), pad="zeros", mode="bilinear").squeeze(0)

        return SDRecPipelineOutput(
            flow_np = flow_pred, # flow
            warped_img = warped_img, # warpped_img
            warped_mask = warped_mask
        )

    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]

        self.empty_text_embed = self.text_encoder(text_input_ids)[0]
        self.empty_text_embed = self.empty_text_embed.to(self.weight_dtype)

        
    @torch.no_grad()
    def single_infer(self,
                     input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,):
        
        if self.normalize_input:
            input_rgb = (input_rgb - 0.5) / 0.5

        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. 
        
        if self.partial_fp16:
            rgb_latent = rgb_latent.to(torch.bfloat16)
        
        # Initial flow map (Guassian noise)
        flow_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.weight_dtype
        )  # [B, 4, H/8, W/8]
        
        # flow_latent = flow_latent.half()
        
        
        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
            
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]
        
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, flow_latent], dim=1
            )  # this order is important: [1,8,H,W]
            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            flow_latent = self.scheduler.step(noise_pred, t, flow_latent).prev_sample
        
        torch.cuda.empty_cache()
        flow = self.decode_flow(flow_latent)

        return flow
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        
        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        
        return rgb_latent
    
    def decode_flow(self, flow_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode flow latent into flow map.

        Args:
            flow_latent (`torch.Tensor`):
                flow latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded flow map.
        """
        # scale latent
        flow_latent = flow_latent / self.latent_scale_factor
        
        # flow_latent = flow_latent.half()
        if self.partial_fp16:
            flow_latent = flow_latent.half()
        else:
            flow_latent = flow_latent.to(self.weight_dtype)
        # decode
        z = self.vae.post_quant_conv(flow_latent)
        stacked = self.vae.decoder(z)

        return stacked


