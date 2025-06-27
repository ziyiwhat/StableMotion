import argparse
import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_ski
from tqdm import tqdm



def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def torch_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results")
    parser.add_argument("--gts")
    args = parser.parse_args()

    gt_path = args.gts
    
    results = glob.glob(os.path.join(args.results, "*.png")) + glob.glob(os.path.join(args.results, "*.jpg"))
    gts = glob.glob(os.path.join(args.gts, "*.jpg")) + glob.glob(os.path.join(args.gts, "*.png"))

    print(f"The test set has {len(results)} images.")

    psnr_ls = []
    ssim_ls = []
    assert len(results) == len(gts), f"results{len(results)} should have the same number of images with gts{len(gts)}"
    
    print("Evaluation process start.")
    with tqdm(
        initial=0,
        total=len(results)
    ) as pbar:
        for i in range(len(results)):
            res = results[i]
            # gt = gts[i]
            base_name = os.path.basename(res)
            img_id = os.path.splitext(base_name)[0]
            gt = f"{gt_path}/{img_id}.jpg"
            assert os.path.exists(gt), f"{gt} do not exists"

            res_img = cv2.imread(res)
            gt_img = cv2.imread(gt)
            _psnr = psnr_ski(gt_img, res_img)
            _ssim = ssim(gt_img, res_img, channel_axis=2)

            psnr_ls.append(_psnr)
            ssim_ls.append(_ssim)

            pbar.set_description(
                f"PSNR: {_psnr:.2f}/{np.mean(psnr_ls):.2f} | SSIM {_ssim:.2f}/{np.mean(ssim_ls):.2f}"
            )
            pbar.update(1)

    print(f"average PSNR is {np.mean(psnr_ls)}")
    print(f"average SSIM is {np.mean(ssim_ls)}")