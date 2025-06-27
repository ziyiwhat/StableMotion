import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from dataloader.dataloader import RecDataset
from torch.utils.data import DataLoader
from dataloader import transforms
import os


# Get Dataset Here
def prepare_dataset(flow_datapath=None,
                    condition_datapath=None,
                    mask_datapath=None,
                    gt_datapath=None,
                    train_flow_list=None,
                    train_condition_list=None,
                    train_mask_list=None,
                    train_gt_list=None,
                    batch_size=1,
                    datathread=4,
                    logger=None,
                    normalized_input=False,
                    flip_input=False,
                    flip_p = 0.5
                    ):
    
    # set the config parameters
    dataset_config_dict = dict()
    train_to_tensor = None
    if not flip_input:
        train_to_tensor = transforms.ToTensor()
    else:
        train_to_tensor = transforms.ToTensorWithFlip(flip_p)
    if not normalized_input:
        train_transform_list = [
                        train_to_tensor,
                        ]
    else:
        train_transform_list = [
                        train_to_tensor,
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]
    train_transform = transforms.Compose(train_transform_list)

    if not normalized_input:
        val_transform_list = [
                        transforms.ToTensor(),
                        ]
    else:
        val_transform_list = [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]
    val_transform = transforms.Compose(val_transform_list)
        
    # TODO: update items in RecDataset
    train_dataset = RecDataset(
        train_flow_dir=flow_datapath,
        train_conditon_dir=condition_datapath, 
        train_mask_dir=mask_datapath, 
        train_gt_dir=gt_datapath,
        train_flow_list=train_flow_list, 
        train_mask_list=train_mask_list, 
        train_gt_list=train_gt_list,
        train_condition_list=train_condition_list, 
        mode='train', 
        transform=train_transform
    )

    img_height, img_width = train_dataset.get_img_size()


    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = True, 
        num_workers = datathread, 
        pin_memory = True
    )

    test_loader = None
    
    num_batches_per_epoch = len(train_loader)
    
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader,test_loader),dataset_config_dict

def Disparity_Normalization(disparity):
    min_value = torch.min(disparity)
    max_value = torch.max(disparity)
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity

def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor