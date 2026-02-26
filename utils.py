from torch_geometric.utils import to_networkx
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import open_clip



def get_full_input():
    full_inputs = []
    with open("input.json", "r") as f:
        input_dict = json.load(f)
        for value in input_dict.values():
            full_inputs.append(get_single_input(value))

    return full_inputs

'''
    "A white object with a frog decoration, placed in the sink with other dishes.": {
    "target": {
      "category": "mug",
      "attribute": ["white", "frog decoration"]
    },
    "tuples": [
      ["frog-decorated mug", "in", "sink"],
      ["frog-decorated mug", "with", "other dishes"]
    ]
    },
'''
def get_single_input(value: dict):
    attributes = ""
    for phrase in value["target"]["attribute"]:
        attributes = attributes + "and" + phrase
    complete_input = attributes + value["target"]["category"]
    return complete_input



def get_correct_num(pred: list[str]):
    correct_list = []
    with open("new.json", "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        for value in json_dict.values():
            print(value)
            correct_list.append(value)

    correct_num = 0
    for i in range(0, len(pred)):
        if pred[i] == correct_list[i]:
            correct_num += 1

    return correct_num

def pick_closest(feature):
    object_feature_dict = get_feature_dict()
    best_label = ""
    best_score = -100.0
    for label, object_feature in object_feature_dict.items():
        curr_score = score(feature, object_feature)
        if curr_score > best_score:
            best_label = label
            best_score = curr_score

    return best_label


def get_clip_feature(text_list):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_type = "ViT-B-16"
    pretrained = "laion2b_s34b_b88k"

    precision = "fp32" if device == "mps" else "fp16"

    model, _, _ = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained,
        precision=precision,
    )
    model = model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(model_type)
    text = tokenizer(text_list).to(device)

    with torch.no_grad():
        feat = model.encode_text(text).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return feat




def show_graph(data):
    G = to_networkx(data, to_undirected=False)

    pos = nx.spring_layout(G, seed=0)

    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20)

    labels = {i: f"{i}\n{data.x[i].tolist()}" for i in range(data.x.size(0))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.axis("off")
    plt.show()

def score(a, b):
    device = "mps" if torch.backends.mps.is_available() else a.device
    a = a.to(device=device, dtype=torch.float32).reshape(1, -1)
    b = b.to(device=device, dtype=torch.float32).reshape(-1)

    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)

    return torch.mm(a, b.unsqueeze(1)).item()


base_path = "/Users/er3r3/Desktop"
data_folder = "/GeoCGA"
data_path = base_path + data_folder
print(os.listdir(data_path))

'''
dict[object_name, object_feature]
'''
def get_feature_dict():
    feature_dict = {}

    for file in os.listdir(data_path):
        object_name = ""
        object_dict = torch.load(os.path.join(data_path, file))
        for i, (k, v) in enumerate(object_dict.items()):
            if i == 0:
                object_name = v
            if i == 1:
                feature_dict[object_name] = v

    return feature_dict


# tensor of all object features [17, 512]
def get_object_features():
    feature_list = []
    feature_dict = get_feature_dict()
    for _, v in feature_dict.items():
        feature_list.append(v)
    return torch.stack(feature_list)


def get_positions(object_names: list[str]):
    """加载 3D 坐标，按 object_names 顺序返回 [N, 3]"""
    data = torch.load("/Users/er3r3/Desktop/position.pth", weights_only=False)
    positions = data["positions"]
    labels = data["labels"]
    label_to_pos = {l: positions[i] for i, l in enumerate(labels)}
    return torch.stack([label_to_pos[n] for n in object_names])


def get_geo_edge_attr(positions, edge_index):
    """几何边特征: [距离(1), 方向(3)] = [E, 4]"""
    src_pos = positions[edge_index[0]]
    dst_pos = positions[edge_index[1]]
    diff = dst_pos - src_pos
    dist = diff.norm(dim=-1, keepdim=True)
    direction = diff / dist.clamp_min(1e-8)
    return torch.cat([dist, direction], dim=-1)


def get_training_samples(object_names: list[str]):
    """
    每个物体 4 条描述，前 3 条 train，第 4 条 test。
    返回 (train_samples, test_samples)。
    每个 sample: (text_feat[1,512], target_idx, tuple_edges)
    tuple_edges: list of (src_idx, dst_idx, relation_feat[512])
    """
    name_to_idx = {name: i for i, name in enumerate(object_names)}

    with open("input.json", "r") as f:
        input_data = json.load(f)
    with open("new.json", "r") as f:
        label_data = json.load(f)

    sentences = list(input_data.keys())

    # 收集所有需要 CLIP 编码的文本
    query_texts = []
    entity_set, relation_set = set(), set()
    for s in sentences:
        info = input_data[s]
        attrs = " ".join(info["target"]["attribute"])
        query_texts.append(f"{attrs} {info['target']['category']}")
        for triple in info.get("tuples", []):
            if len(triple) == 3:
                entity_set.add(triple[0])
                entity_set.add(triple[2])
                relation_set.add(triple[1])

    entity_list = sorted(entity_set)
    relation_list = sorted(relation_set)

    # 一次 CLIP 调用编码所有文本
    all_texts = query_texts + list(object_names) + entity_list + relation_list
    all_feats = get_clip_feature(all_texts).cpu()

    n_q, n_o, n_e = len(query_texts), len(object_names), len(entity_list)
    query_feats = all_feats[:n_q]
    obj_name_feats = all_feats[n_q:n_q + n_o]
    entity_feats = all_feats[n_q + n_o:n_q + n_o + n_e]
    relation_feats = all_feats[n_q + n_o + n_e:]

    relation_to_feat = {r: relation_feats[i] for i, r in enumerate(relation_list)}

    # 用 CLIP 相似度把 tuple 实体匹配到 17 个物体
    sim_matrix = entity_feats @ obj_name_feats.T
    best_scores, best_indices = sim_matrix.max(dim=1)

    THRESHOLD = 0.7
    entity_to_obj_idx = {}
    for i, name in enumerate(entity_list):
        if best_scores[i].item() >= THRESHOLD:
            entity_to_obj_idx[name] = best_indices[i].item()

    print(f"Entity matching ({len(entity_to_obj_idx)}/{len(entity_list)}):")
    for ent, idx in sorted(entity_to_obj_idx.items()):
        print(f"  '{ent}' -> '{object_names[idx]}'")

    # 构建 samples
    train_samples, test_samples = [], []
    total_tuples, matched_tuples = 0, 0
    for i, s in enumerate(sentences):
        label = label_data[s]
        if label not in name_to_idx:
            continue

        target_idx = name_to_idx[label]
        tuple_edges = []
        for triple in input_data[s].get("tuples", []):
            if len(triple) != 3:
                continue
            total_tuples += 1
            ent_a, rel, ent_b = triple
            if ent_a in entity_to_obj_idx and ent_b in entity_to_obj_idx:
                src = entity_to_obj_idx[ent_a]
                dst = entity_to_obj_idx[ent_b]
                if src != dst:
                    matched_tuples += 1
                    tuple_edges.append((src, dst, relation_to_feat[rel]))

        sample = (query_feats[i:i + 1], target_idx, tuple_edges)
        if i % 4 == 3:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    print(f"Tuples matched to object pairs: {matched_tuples}/{total_tuples}")
    return train_samples, test_samples




# data_dict = {}
#
# with open("../预处理/frame_00010.json", "r", encoding="utf-8") as f:
#     json_str = f.read()
#     json_dict = json.loads(json_str)
#     sum_object = 0
#     sum_sample = 0
#     for each in json_dict["object"]:
#         sum_object += 1
#         for sentence in each["sentence"]:
#             sum_sample += 1
#             data_dict[sentence] = each["category"]
#
# with open("new.json", "w") as f:
#     json.dump(data_dict, f, indent=2)
