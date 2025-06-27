import cv2
import numpy as np
from tqdm import tqdm

ensemble_size = 2
img_number = 519
threshold = 240
# img_number = 5
source_base_template = "/home/wangzhangcheng/share-60/output_randomTimesteps_wSD_checker/70000-{}/"
target_path_mix = "/home/wangzhangcheng/share-60/output_randomTimesteps_wSD_checker/70000-ensemble8/"
preset_mask = np.zeros((384,512), dtype=np.bool_)
corner_size = 20
preset_mask[:corner_size, :] = True
preset_mask[-corner_size:, :] = True
preset_mask[:, :corner_size] = True
preset_mask[:, -corner_size:] = True

for i in tqdm(range(1, img_number+1)):
    ind = str(i).zfill(5)
    img_name = ind + ".png"
    imgs = []
    for j in range(ensemble_size):
        base_dir = source_base_template.format(j)
        img_path = base_dir + img_name

        img = cv2.imread(img_path)
        imgs.append(img)
    # mask 计算
    mask_union_any = preset_mask

    # img 计算
    imgs = [
        i[None] for i in imgs
    ]
    img_union = np.concatenate(imgs, axis=0)
    img_union_min = np.min(img_union, axis=0)
    img_union_median = np.median(img_union, axis=0).astype(np.uint8)
    # print("min:", img_union_min.shape, img_union_min.dtype)
    # print("median:", img_union_median.shape, img_union_median.dtype)

    img_union_mix = img_union_median.copy()
    img_union_mix[mask_union_any] = img_union_min[mask_union_any]
    # print("mix:", img_union_mix.shape, img_union_mix.dtype)
    cv2.imwrite(target_path_mix + img_name, img_union_mix)
