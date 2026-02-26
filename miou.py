import os
import torch
import numpy as np
from PIL import Image

def calculate_iou(pred_mask, gt_mask):
    intersection = torch.logical_and(pred_mask, gt_mask).sum().float()
    union = torch.logical_or(pred_mask, gt_mask).sum().float()
    if union == 0:  # 两个都是空mask时避免除0（你也可以改成返回0.0）
        return 1.0
    iou = intersection / union
    return iou.item()

def load_mask(file_path):
    mask = Image.open(file_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float()
    mask = mask > 0
    return mask

def calculate_iou_for_directory(render_dir, gt_dir):
    iou_list = []

    for filename in os.listdir(render_dir):
        if filename.endswith(".png"):
            render_path = os.path.join(render_dir, filename)
            gt_path = os.path.join(gt_dir, filename)

            if not os.path.exists(gt_path):
                continue  # gt里没有同名文件就跳过（你也可以改成raise）
            pred_mask = load_mask(render_path)
            gt_mask = load_mask(gt_path)

            if gt_mask.sum() == 0:
                continue

            iou = calculate_iou(pred_mask, gt_mask)
            iou_list.append(iou)

    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    return mean_iou

def miou_single_file(render_file, gt_file):
    pred_mask = load_mask(render_file)
    gt_mask = load_mask(gt_file)
    if gt_mask.sum() == 0:
        return "gt mask is 0"
    return calculate_iou(pred_mask, gt_mask)

def compute_iou_auto(render_path, gt_path):
    """
    - render_path 是文件：算单张 IoU
      - gt_path 也是文件：直接用
      - gt_path 是文件夹：自动用同名文件 (gt_path/basename(render_path))
    - render_path 是文件夹：算目录平均 IoU（要求 gt_path 也是文件夹）
    """
    if not os.path.exists(render_path):
        raise FileNotFoundError(f"render_path not found: {render_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_path not found: {gt_path}")

    # render 是文件
    if os.path.isfile(render_path):
        if os.path.isdir(gt_path):
            gt_path = os.path.join(gt_path, os.path.basename(render_path))
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"gt file not found: {gt_path}")
        elif not os.path.isfile(gt_path):
            raise ValueError("gt_path must be a file or directory.")
        return miou_single_file(render_path, gt_path)

    # render 是文件夹
    if os.path.isdir(render_path):
        if not os.path.isdir(gt_path):
            raise ValueError("When render_path is a directory, gt_path must also be a directory.")
        return calculate_iou_for_directory(render_path, gt_path)

    raise ValueError("render_path must be a file or directory.")

# ======================
# 用法示例
# ======================
render_path = "/root/autodl-tmp/Dr-Splat/outputs_1_pq_openclip_topk45_weight_128/train/renders_mask_toaster/00089.png"  # 可以填文件或文件夹
gt_path = "/root/autodl-tmp/Ref-Lerf/waldo_kitchen/gt_mask/frame_00090/toaster.png"      # 可以填文件或文件夹（见上面规则）

iou_value = compute_iou_auto(render_path, gt_path)
print(f"IoU/mIoU: {iou_value}")
