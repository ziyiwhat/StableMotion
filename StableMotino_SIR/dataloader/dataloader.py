from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms as T

from PIL import Image
from skimage import io, transform
import numpy as np

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img

class RecDataset(Dataset):
    def __init__(self, 
                 train_flow_dir,
                 train_conditon_dir,
                 train_mask_dir, 
                 train_gt_dir,
                 train_flow_list,
                 train_condition_list,
                 train_mask_list,
                 train_gt_list,
                 dataset_name='DIR-D',
                 mode='train',
                 save_filename=False,
                 transform=None):
        super(RecDataset, self).__init__()

        self.train_flow_dir = train_flow_dir
        self.train_condition_dir = train_conditon_dir
        self.train_mask_dir = train_mask_dir
        self.train_gt_dir = train_gt_dir

        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        self.train_flow_list = train_flow_list
        self.train_condition_list = train_condition_list
        self.train_mask_list = train_mask_list
        self.train_gt_list = train_gt_list

        # 输入图像的大小
        self.img_size=(512, 384)
        

        DIRD_finalpass_dict = {
            'train_flow':  self.train_flow_list,
            'train_cond': self.train_condition_list, 
            'train_mask': self.train_mask_list,
            'train_gt': self.train_gt_list
        }

        dataset_name_dict = {
            'DIR-D': DIRD_finalpass_dict
        }

        self.samples = []

        key_flow = self.mode + '_flow'
        key_rgb  = self.mode + '_cond'
        key_mask = self.mode + '_mask'
        key_gt = self.mode + '_gt'
        
        flow_filenames = dataset_name_dict[dataset_name][key_flow]
        cond_filenames = dataset_name_dict[dataset_name][key_rgb]
        mask_filenames = dataset_name_dict[dataset_name][key_mask]
        gt_filenames = dataset_name_dict[dataset_name][key_gt]

        lines_flow = read_text_lines(flow_filenames)
        lines_cond = read_text_lines(cond_filenames)
        lines_mask = read_text_lines(mask_filenames)
        lines_gt = read_text_lines(gt_filenames)

        for line_flow, line_cond, line_mask, line_gt in zip(lines_flow, lines_cond, lines_mask, lines_gt):

            assert line_cond.split(".")[0] == line_flow.split(".")[0], "unmatched data pairs"

            sample = dict()

            sample['cond'] = os.path.join(train_conditon_dir, line_cond)
            sample['flow'] = os.path.join(train_flow_dir, line_flow)
            sample['mask'] = os.path.join(train_mask_dir, line_mask)
            sample['gt'] = os.path.join(train_gt_dir, line_gt)

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        sample['cond'] = read_img(sample_path['cond'])  # [H, W, 3]
        sample['mask'] = read_img(sample_path['mask'])
        sample['gt'] = read_img(sample_path['gt'])

        trans = T.ToTensor()
        sample['flow'] = np.load(sample_path['flow'])
        sample['flow'] = trans(sample['flow'])
        zeros = torch.ones_like(sample['flow'])
        zeros = zeros.chunk(2, dim=0)[0]
        sample['flow'] = torch.cat((sample['flow'], zeros), dim=0)
        sample['flow'] = sample['flow'] / 50 # vital factor

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size
