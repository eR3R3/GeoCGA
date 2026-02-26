import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import utils


def build_fc_edge_index(num_nodes):
    """全连接图，无自环"""
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]])


class SimpleEdgeMP(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, node_dim), nn.ReLU())
        self.upd_mlp = nn.Sequential(nn.Linear(node_dim * 2, node_dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1)) + x


class RefGNN(nn.Module):
    def __init__(self, node_dim=513, edge_dim=516, tau=0.1):
        super().__init__()
        self.conv = SimpleEdgeMP(node_dim, edge_dim)
        self.head = nn.Linear(node_dim, 1)
        self.geo_proj = nn.Linear(4, 512)
        self.tau = tau

    def forward(self, visual_feats, text_feat, edge_index, edge_attr):
        sim = F.cosine_similarity(visual_feats, text_feat.expand_as(visual_feats), dim=-1, eps=1e-12)
        x = torch.cat([visual_feats, sim.unsqueeze(-1)], dim=-1)  # [N, 513]
        x = self.conv(x, edge_index, edge_attr)
        scores = self.head(x).squeeze(-1) / self.tau  # [N]
        return scores

    def relation_loss(self, geo_edge_attr, tuple_edges, edge_to_idx):
        """relation-level alignment: 文本关系 CLIP feat 和投影后的几何边要相似"""
        if not tuple_edges:
            return torch.tensor(0.0, device=geo_edge_attr.device)
        rel_feats, geo_feats = [], []
        for src, dst, rel_feat in tuple_edges:
            eidx = edge_to_idx.get((src, dst))
            if eidx is not None:
                rel_feats.append(rel_feat)
                geo_feats.append(geo_edge_attr[eidx])
        if not rel_feats:
            return torch.tensor(0.0, device=geo_edge_attr.device)
        rel_feats = torch.stack(rel_feats).to(geo_edge_attr.device)   # [K, 512]
        geo_feats = self.geo_proj(torch.stack(geo_feats))             # [K, 512]
        return -F.cosine_similarity(rel_feats, geo_feats, dim=-1).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 17 个物体的 CLIP 视觉特征 + 3D 位置
    feature_dict = utils.get_feature_dict()
    object_names = list(feature_dict.keys())
    visual_feats = torch.stack([feature_dict[n] for n in object_names]).float().to(device)
    N = len(object_names)

    edge_index = build_fc_edge_index(N).to(device)
    E = edge_index.size(1)

    # (src, dst) → edge position 的映射
    edge_to_idx = {}
    for e in range(E):
        s, d = edge_index[0, e].item(), edge_index[1, e].item()
        edge_to_idx[(s, d)] = e

    # 几何边特征 [E, 4]，对所有 sample 固定不变
    positions = utils.get_positions(object_names).to(device)
    geo_edge_attr = utils.get_geo_edge_attr(positions, edge_index)  # [E, 4]

    train_samples, test_samples = utils.get_training_samples(object_names)
    print(f"{N} objects, {len(train_samples)} train, {len(test_samples)} test")

    model = RefGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_rel = 0.1

    for epoch in range(200):
        # --- train ---
        model.train()
        total_loss, correct = 0.0, 0
        for text_feat, target_idx, tuple_edges in train_samples:
            text_feat = text_feat.to(device)
            target = torch.tensor([target_idx], device=device)

            # 构建本 sample 的 edge_attr: [tuple_clip(512) | geo(4)] = [E, 516]
            tuple_feat = torch.zeros(E, 512, device=device)
            for src, dst, rel_feat in tuple_edges:
                eidx = edge_to_idx.get((src, dst))
                if eidx is not None:
                    tuple_feat[eidx] = rel_feat.to(device)
            edge_attr = torch.cat([tuple_feat, geo_edge_attr], dim=-1)  # [E, 516]

            scores = model(visual_feats, text_feat, edge_index, edge_attr)
            node_loss = F.cross_entropy(scores.unsqueeze(0), target)
            rel_loss = model.relation_loss(geo_edge_attr, tuple_edges, edge_to_idx)
            loss = node_loss + lambda_rel * rel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if scores.argmax().item() == target_idx:
                correct += 1

        # --- test ---
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for text_feat, target_idx, tuple_edges in test_samples:
                text_feat = text_feat.to(device)
                tuple_feat = torch.zeros(E, 512, device=device)
                for src, dst, rel_feat in tuple_edges:
                    eidx = edge_to_idx.get((src, dst))
                    if eidx is not None:
                        tuple_feat[eidx] = rel_feat.to(device)
                edge_attr = torch.cat([tuple_feat, geo_edge_attr], dim=-1)

                scores = model(visual_feats, text_feat, edge_index, edge_attr)
                if scores.argmax().item() == target_idx:
                    test_correct += 1

        if (epoch + 1) % 20 == 0:
            train_acc = correct / len(train_samples)
            test_acc = test_correct / len(test_samples)
            print(f"Epoch {epoch+1:3d} | Loss {total_loss/len(train_samples):.4f} | Train {train_acc:.2%} | Test {test_acc:.2%}")
