'''
Originated from Dr-Splat Repository 
Yuwen Tao 11/5 2025
'''

import numpy as np
import torch
from scene import Scene
import os

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, count_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path

from evaluation.openclip_encoder import OpenCLIPNetwork

import time

import faiss

from evaluation import colormaps

import cv2

COLORMAP_OPTIONS = colormaps.ColormapOptions(
    colormap="turbo",
    normalize=True,
    colormap_min=-1.0,
    colormap_max=1.0,
)

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, label, clip_model, img_label, activation_features, threshold, mask_percentile=0.90):
    mask_path = os.path.join(model_path, name, f"renders_mask_{img_label}")
    makedirs(mask_path, exist_ok=True)

    gaussian_activations = activation_features.squeeze(1)  # [N] - 每个Gaussian的激活度

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = count_render(view, gaussians, pipeline, background)
        ids = render_pkg['per_pixel_gaussian_ids'].detach()  # [H, W, top_k]
        contributions = render_pkg['per_pixel_gaussian_contributions'].detach()  # [H, W, top_k]
        
        H, W = ids.shape[:2]
        top_k = ids.shape[2]
        
        ids_flat = ids.reshape(-1, top_k).long()  # [H*W, top_k]
        contribs_flat = contributions.reshape(-1, top_k)  # [H*W, top_k]
        
        valid_mask = ids_flat != -1  # [H*W, top_k]
        
        contribs_normalized = contribs_flat.clone()
        contribs_normalized[~valid_mask] = 0
        contribs_sum = contribs_normalized.sum(dim=1, keepdim=True)  # [H*W, 1]
        contribs_normalized = contribs_normalized / (contribs_sum + 1e-9)
        
        pixel_activations = torch.zeros(H * W, device="cuda")
        for k in range(top_k):
            valid_pixels = valid_mask[:, k]
            if valid_pixels.sum() > 0:
                pixel_ids = ids_flat[valid_pixels, k]
                pixel_weights = contribs_normalized[valid_pixels, k]
                pixel_activations[valid_pixels] += gaussian_activations[pixel_ids] * pixel_weights
        pixel_activations = pixel_activations.reshape(H, W)
        if threshold > 0:
            mask_threshold = threshold
        else:
            valid_activations = pixel_activations[pixel_activations > 0]
            if len(valid_activations) > 0:
                mask_threshold = torch.quantile(valid_activations, mask_percentile)
            else:
                mask_threshold = 0.0
        mask = (pixel_activations > mask_threshold).float() * 255.0
        mask_np = mask.cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(mask_path, f"{idx:05d}.png"), mask_np)


def get_activation_score(dataset : ModelParams, args, label, clip_model, index, return_mean_features=False):
    """
    获取activation score
    
    Args:
        dataset: 模型参数
        args: 命令行参数
        label: 标签索引
        clip_model: CLIP模型
        index: FAISS索引
        return_mean_features: 是否返回mean_features
        
    Returns:
        如果 return_mean_features=False:
            gaussians, scene, activation_features, background
        如果 return_mean_features=True:
            gaussians, scene, activation_features, background, mean_features, mean_position
    """
    with torch.no_grad():
        start_time = time.time()
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')

        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # features: [gaussian_num, language_feature_dim]
        features = gaussians._language_feature.clone()

        # 找出 features 里"最后一维上所有元素都等于 -1"的那些位置，并把这些位置标成 True
        # zero_mask: [gaussian_num]
        zero_mask = torch.all(features == -1, dim=-1)

        # 把 features 中 zero_mask 为 False 的位置对应的特征拿出来
        # leaf_lang_feat: [num_non_zero_features, language_feature_dim]
        leaf_lang_feat = torch.from_numpy(index.sa_decode(features[~zero_mask].cpu().numpy())).to("cuda")

        # activation_features: [gaussian_num, 1]
        activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
        # _activation_features: [num_non_zero_features, 1], 每个gaussian的分数
        _activation_features = clip_model.get_activation(leaf_lang_feat, label)
        # activation_features: [gaussian_num, 1], 每个gaussian的分数
        activation_features[~zero_mask] = _activation_features

        thr = args.threshold
        
        # squeeze把activation_features变成一维的tensor，然后找到大于thr的位置
        # activation_threshold: [num_activated_features], 存储的是满足条件的gaussian的索引位置
        activation_threshold = torch.where(activation_features.squeeze() > thr)[0]
        print(f"Activation threshold indices (first 10): {activation_threshold[0:10]}")

        # 计算被选中的 gaussians 的 mean features
        # 首先，创建一个全体 gaussians 的解码后 features tensor
        # decoded_features: [gaussian_num, language_feature_dim]
        decoded_features = torch.zeros((features.shape[0], leaf_lang_feat.shape[1]), dtype=torch.float32).cuda()
        # 将非零部分填充为解码后的 features
        decoded_features[~zero_mask] = leaf_lang_feat
        
        # 获取被选中的 gaussians 的 features
        # selected_features: [num_activated_features, language_feature_dim]
        selected_features = decoded_features[activation_threshold]
        
        # 计算平均 features
        # mean_features: [language_feature_dim]
        mean_features = selected_features.mean(dim=0)
        
        # 计算被选中的 gaussians 的 mean position
        # positions: [gaussian_num, 3]
        positions = gaussians.get_xyz
        # selected_positions: [num_activated_features, 3]
        selected_positions = positions[activation_threshold]
        # mean_position: [3]
        mean_position = selected_positions.mean(dim=0)
        
        print(f"Number of selected gaussians: {len(activation_threshold)}")
        print(f"Mean features shape: {mean_features.shape}")
        print(f"Mean features (first 10 dims): {mean_features[:10]}")
        print(f"Mean position: {mean_position}")

        # 计算 mean_features 与各个物体词的相似度分数
        object_labels = [
            "knife",
            "ottolenghi",
            "pour-over vessel",
            "stainless steel pot",
            "toaster",
            "yellow desk",
            "plastic ladle",
            "pot",
            "refrigerator",
            "spatula",
            "cabinet",
            "ketchup",
            "dark cup",
            "frog cup",
            "plate",
            "black spoon",
            "red cup"
        ]
        
        # 让所有物体标签变成positives
        clip_model.set_positives(object_labels)
        
        print("\n" + "="*60)
        print("Relevancy scores between mean_features and object labels:")
        print("="*60)
        
        # get_relevancy 可以计算每个物体的分数
        # mean_features 需要是 [1, feature_dim] 的形状
        mean_features_batch = mean_features.unsqueeze(0)  # [1, feature_dim]
        
        for idx, obj_label in enumerate(object_labels):
            # get_relevancy 计算相关性分数
            # positive_id 是当前物体在 positives 列表中的索引
            probs = clip_model.get_relevancy(mean_features_batch, idx)
            # probs: [1, 2], 第一列是正类概率，第二列是负类概率
            pos_prob = probs[0, 0].item()  # 正类概率
            print(f"{obj_label:25s}: {pos_prob:.4f}")
        
        print("="*60 + "\n")

        end_time = time.time()
        print(f'Get activation score time: {end_time - start_time}s')
        
        if return_mean_features:
            return gaussians, scene, activation_features, background, mean_features, mean_position
        else:
            return gaussians, scene, activation_features, background


def render_activation_score(dataset : ModelParams, pipeline : PipelineParams, gaussians, scene, 
                            activation_features, background, args, label, clip_model, 
                            img_save_label, skip_train : bool, skip_test : bool):
    
    with torch.no_grad():
        start_time = time.time()
        
        thr = args.threshold

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, 
                       scene.getTrainCameras(), gaussians, pipeline, background, args, 
                       label, clip_model, img_save_label, activation_features, thr, args.mask_percentile)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, 
                       scene.getTestCameras(), gaussians, pipeline, background, args, 
                       label, clip_model, img_save_label, activation_features, thr, args.mask_percentile)
        
        end_time = time.time()
        print(f'time: {end_time - start_time}s')


def save_precomputed_features(object_label, mean_features, mean_position, activation_features, save_dir="precomputed_features"):
    os.makedirs(save_dir, exist_ok=True)
    
    safe_label = object_label.replace(" ", "_").replace("/", "-")
    save_path = os.path.join(save_dir, f"{safe_label}.pth")
    
    torch.save({
        'object_label': object_label,
        'mean_features': mean_features.cpu(),
        'mean_position': mean_position.cpu(),
        'activation_features': activation_features.cpu()
    }, save_path)
    
    print(f"Saved precomputed features for '{object_label}' to {save_path}")
    print(f"  Mean position: {mean_position.cpu()}")


def load_precomputed_features(save_dir="precomputed_features"):
    precomputed_data = {}
    
    if not os.path.exists(save_dir):
        print(f"Precomputed features directory '{save_dir}' does not exist.")
        return precomputed_data
    
    for filename in os.listdir(save_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(save_dir, filename)
            data = torch.load(filepath)
            object_label = data['object_label']
            precomputed_data[object_label] = {
                'mean_features': data['mean_features'].cuda(),
                'mean_position': data['mean_position'].cuda() if 'mean_position' in data else None,
                'activation_features': data['activation_features'].cuda()
            }
            pos_str = f", pos={data['mean_position']}" if 'mean_position' in data else ""
            print(f"Loaded precomputed features for '{object_label}'{pos_str}")
    
    return precomputed_data


def find_best_matching_object(query_text, precomputed_data, clip_model):
    # 编码查询文本
    query_embedding = clip_model.encode_text(query_text, device="cuda").float()
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    
    best_score = -1.0
    best_object_label = None
    best_activation_features = None
    
    print("\n" + "="*60)
    print(f"Matching query '{query_text}' with precomputed objects:")
    print("="*60)
    
    for object_label, data in precomputed_data.items():
        mean_features = data['mean_features']
        # 归一化
        mean_features_normalized = mean_features / mean_features.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        similarity = torch.mm(query_embedding, mean_features_normalized.unsqueeze(1)).item()
        
        print(f"{object_label:25s}: {similarity:.4f}")
        
        if similarity > best_score:
            best_score = similarity
            best_object_label = object_label
            best_activation_features = data['activation_features']
    
    print("="*60)
    print(f"Best match: '{best_object_label}' (score: {best_score:.4f})")
    print("="*60 + "\n")
    
    return best_object_label, best_score, best_activation_features


def precompute_all_objects(dataset : ModelParams, args, clip_model, index, object_labels, save_dir="precomputed_features"):
    print("\n" + "="*60)
    print("Precomputing features for all objects...")
    print("="*60 + "\n")
    
    all_positions = []
    
    for idx, object_label in enumerate(object_labels):
        print(f"\n[{idx+1}/{len(object_labels)}] Processing: {object_label}")
        print("-" * 60)
        
        # 设置当前物体为 positive
        clip_model.set_positives([object_label])
        
        # 计算 activation score 和 mean_features（label=0 因为只有一个positive）
        gaussians, scene, activation_features, background, mean_features, mean_position = get_activation_score(
            dataset, args, label=0, clip_model=clip_model, index=index, return_mean_features=True
        )
        
        all_positions.append(mean_position)
        
        # 保存数据
        save_precomputed_features(object_label, mean_features, mean_position, activation_features, save_dir)
    
    # 汇总所有物体的 mean position 到一个文件: [17, 3]
    all_positions_tensor = torch.stack(all_positions, dim=0)  # [num_objects, 3]
    position_save_path = os.path.join(save_dir, "position.pth")
    torch.save({
        'positions': all_positions_tensor.cpu(),
        'labels': object_labels,
    }, position_save_path)
    
    print("\n" + "="*60)
    print(f"Saved all mean positions to {position_save_path}")
    print(f"  Shape: {all_positions_tensor.shape}")
    for i, label in enumerate(object_labels):
        print(f"  {label:25s}: {all_positions_tensor[i].cpu().tolist()}")
    print("="*60)
    
    print("\n" + "="*60)
    print("Precomputation completed!")
    print("="*60 + "\n")


def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, label, clip_model, index, img_save_label):
    # 获取activation score
    gaussians, scene, activation_features, background = get_activation_score(
        dataset, args, label, clip_model, index
    )
    
    # 渲染activation score
    render_activation_score(
        dataset, pipeline, gaussians, scene, activation_features, background,
        args, label, clip_model, img_save_label, skip_train, skip_test
    )


def render_from_query(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                      args, query_text, clip_model, precomputed_data):
    # 找到最匹配的物体
    best_object_label, best_score, activation_features = find_best_matching_object(
        query_text, precomputed_data, clip_model
    )
    
    if best_object_label is None:
        print("No matching object found!")
        return
    
    # 重新加载模型（因为需要 gaussians 和 scene）
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        
        checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 使用最匹配物体的 activation_features 进行渲染
    # 将 query_text 作为 img_save_label
    img_save_label = query_text.replace(" ", "_")
    
    render_activation_score(
        dataset, pipeline, gaussians, scene, activation_features, background,
        args, label=0, clip_model=clip_model, img_save_label=img_save_label, 
        skip_train=skip_train, skip_test=skip_test
    )


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--semantic_model", default='dino', type=str)
    parser.add_argument("--pq_index", type=str, default=None)
    parser.add_argument("--img_save_label", type=str, default=None)
    parser.add_argument("--img_label", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--mask_percentile", type=float, default=0.90, help="Percentile threshold for mask (0.90 = keep top 10% of activations)")
    parser.add_argument("--precompute", action="store_true", help="Precompute features for all objects")
    parser.add_argument("--query", type=str, default="", help="Query text to find matching object and render")
    parser.add_argument("--precompute_dir", type=str, default="precomputed_features_waldo", help="Directory to save/load precomputed features")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    
    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    index = faiss.read_index(args.pq_index)

    # these are all objects for datasets kitchen, the masks are segmented using 2D foundation model Grounded-Sam
    # for more information, checkout paper ReferSplat and their dataset Ref-Lerf
    all_object_labels = [
        "knife",
        "ottolenghi",
        "pour-over vessel",
        "stainless steel pot",
        "toaster",
        "yellow desk",
        "plastic ladle",
        "pot",
        "refrigerator",
        "spatula",
        "cabinet",
        "ketchup",
        "dark cup",
        "frog cup",
        "plate",
        "black spoon",
        "red cup"
    ]

    print("\n".join([f"{k} = {v}" for k, v in sorted(vars(args).items())]))
    
    print(args.precompute)
    print(args.query)
    # if not precomputed yet, do precompute, the clustering happens here
    if args.precompute:
        print("\n" + "="*60)
        print("precomput")
        print("="*60)
        precompute_all_objects(
            model.extract(args), args, clip_model, index, 
            all_object_labels, args.precompute_dir
        )
    
    # if already precomputed, render based on query
    elif args.query != "":
        print("\n" + "="*60)
        print(f"rendering using query")
        print(f"Query: '{args.query}'")
        print("="*60)
        
        # 加载precompute数据
        precomputed_data = load_precomputed_features(args.precompute_dir)
        
        if not precomputed_data:
            print("\nError: No precomputed features found!")
            print(f"Please run with --precompute first to generate features in '{args.precompute_dir}'")
        else:
            render_from_query(
                model.extract(args), pipeline.extract(args), 
                args.skip_train, args.skip_test, args, 
                args.query, clip_model, precomputed_data
            )
    
    # using original way, which is selecting gaussians to render
    elif args.img_label is not None:
        print("\n" + "="*60)
        print(f"original rendering")
        print(f"Object: '{args.img_label}'")
        print("="*60)
        
        img_labels = [args.img_label]
        clip_model.set_positives(img_labels)
        
        for label in range(len(img_labels)):
            text_feat = clip_model.encode_text(img_labels[label], device=device).float()
            render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test, args, label, clip_model, index, args.img_save_label)
    
    else:
        print("\nError: Please specify one of the following modes:")
        print("  --precompute              : Precompute features for all objects")
        print("  --query 'text'            : Render based on query text")
        print("  --img_label 'object'      : Render a specific object (legacy mode)")

