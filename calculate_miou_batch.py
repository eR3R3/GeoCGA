#!/usr/bin/env python3
"""
Note: This file is completely written using GPT-5
批量计算mIoU的脚本

用法:
    python calculate_miou_batch.py <folder1> <folder2> <name>

参数:
    folder1: 包含png文件的目录（按顺序遍历）
    folder2: 包含子文件夹的目录，每个子文件夹可能包含name.png
    name: 要查找的文件名（如 "toaster.png" 或 "toaster"）

功能:
    - 遍历folder1中的每个png文件（按文件名排序）
    - 对于每个png，找到folder2中对应顺序的子文件夹
    - 在子文件夹中查找name.png（如果name不包含扩展名，自动添加.png）
    - 如果找到name.png，调用miou_single_file计算IoU
    - 如果没找到，跳过该对
    - 最后计算所有IoU的平均值（mean IoU）
"""

import os
import sys
import numpy as np
from miou import miou_single_file

def get_sorted_pngs(folder):
    """获取文件夹中按文件名排序的png文件列表"""
    png_files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    png_files.sort()
    return [os.path.join(folder, f) for f in png_files]

def get_sorted_subfolders(folder):
    """获取文件夹中按名称排序的子文件夹列表"""
    subfolders = [f for f in os.listdir(folder) 
                  if os.path.isdir(os.path.join(folder, f))]
    subfolders.sort()
    return [os.path.join(folder, f) for f in subfolders]

def normalize_name(name):
    """规范化文件名，如果没有扩展名则添加.png"""
    if not name.lower().endswith('.png'):
        return name + '.png'
    return name

def calculate_batch_miou(folder1, folder2, name):
    """
    批量计算mIoU
    
    Args:
        folder1: 包含png文件的目录
        folder2: 包含子文件夹的目录
        name: 要查找的文件名
    
    Returns:
        mean_iou: 平均IoU值
        iou_list: 所有IoU值的列表
        processed_count: 成功处理的文件对数
        skipped_count: 跳过的文件对数
    """
    # 规范化文件名
    name = normalize_name(name)
    
    # 获取排序后的文件列表
    png_files = get_sorted_pngs(folder1)
    subfolders = get_sorted_subfolders(folder2)
    
    print(f"Folder1中找到 {len(png_files)} 个PNG文件")
    print(f"Folder2中找到 {len(subfolders)} 个子文件夹")
    
    if len(png_files) != len(subfolders):
        print(f"警告: PNG文件数量({len(png_files)})与子文件夹数量({len(subfolders)})不一致!")
    
    iou_list = []
    processed_count = 0
    skipped_count = 0
    
    # 按顺序遍历
    min_count = min(len(png_files), len(subfolders))
    
    for i in range(min_count):
        png_file = png_files[i]
        subfolder = subfolders[i]
        
        # 查找name.png
        name_path = os.path.join(subfolder, name)
        
        if not os.path.exists(name_path):
            print(f"跳过: {os.path.basename(png_file)} - 子文件夹 {os.path.basename(subfolder)} 中没有 {name}")
            skipped_count += 1
            continue
        
        # 计算IoU
        try:
            iou_result = miou_single_file(png_file, name_path)
            
            # 处理返回值可能是字符串的情况（gt mask is 0）
            if isinstance(iou_result, str):
                print(f"跳过: {os.path.basename(png_file)} - {iou_result}")
                skipped_count += 1
                continue
            
            iou_list.append(iou_result)
            processed_count += 1
            print(f"处理: {os.path.basename(png_file)} <-> {os.path.basename(subfolder)}/{name} : IoU = {iou_result:.4f}")
            
        except Exception as e:
            print(f"错误: 处理 {os.path.basename(png_file)} 时出错: {e}")
            skipped_count += 1
            continue
    
    # 计算平均IoU
    if iou_list:
        mean_iou = float(np.mean(iou_list))
    else:
        mean_iou = 0.0
    
    return mean_iou, iou_list, processed_count, skipped_count

def main():
    if len(sys.argv) != 4:
        print("用法: python calculate_miou_batch.py <folder1> <folder2> <name>")
        print("\n参数说明:")
        print("  folder1: 包含png文件的目录（按顺序遍历）")
        print("  folder2: 包含子文件夹的目录，每个子文件夹可能包含name.png")
        print("  name: 要查找的文件名（如 'toaster.png' 或 'toaster'）")
        sys.exit(1)
    
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    name = sys.argv[3]
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder1):
        print(f"错误: folder1不存在或不是目录: {folder1}")
        sys.exit(1)
    
    if not os.path.isdir(folder2):
        print(f"错误: folder2不存在或不是目录: {folder2}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Folder1: {folder1}")
    print(f"Folder2: {folder2}")
    print(f"查找文件: {normalize_name(name)}")
    print("=" * 60)
    print()
    
    # 计算mIoU
    mean_iou, iou_list, processed_count, skipped_count = calculate_batch_miou(folder1, folder2, name)
    
    # 输出结果
    print()
    print("=" * 60)
    print("结果统计:")
    print(f"  成功处理: {processed_count} 对文件")
    print(f"  跳过: {skipped_count} 对文件")
    print(f"  平均IoU (mIoU): {mean_iou:.6f}")
    if iou_list:
        print(f"  IoU范围: [{min(iou_list):.6f}, {max(iou_list):.6f}]")
    print("=" * 60)
    
    # 输出mIoU值到标准输出（单独一行，方便脚本解析）
    print(f"MIOU_VALUE: {mean_iou:.6f}")

if __name__ == "__main__":
    main()
