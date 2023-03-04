import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]

    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


def positional_encoding(d_model, length):
    pos = torch.arange(0, length).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    angle_rads = pos * torch.exp(i * -(math.log(10000) / d_model))
    pe = torch.zeros(1, length, d_model)
    pe[0, :, 0::2] = torch.sin(angle_rads)
    pe[0, :, 1::2] = torch.cos(angle_rads)
    return pe


def split_train_test(num_samples, batch_size, graph_steps, train_ratio=0.5):
    idx = np.arange(num_samples)
    max_idx = int(train_ratio * num_samples)
    train_idx = idx[:max_idx]
    test_idx = idx[max_idx:]
    if train_ratio == 1.0:
        train_idx = idx
        test_idx = idx

    train_idx_to_save = torch.randperm(train_idx.max() - (graph_steps - 1) - 1)
    if train_ratio == 1.0:
        test_idx_to_save = torch.arange(test_idx.max() - (graph_steps - 1) - 1)
    else:
        test_idx_to_save = torch.arange(max_idx, test_idx.max() - (graph_steps - 1) - 1)

    train_idx = TensorDataset(train_idx_to_save)
    train_idx = DataLoader(train_idx, batch_size=batch_size)

    test_idx = TensorDataset(test_idx_to_save)
    test_idx = DataLoader(test_idx, batch_size=batch_size)

    return train_idx, test_idx, train_idx_to_save.numpy(), test_idx_to_save.numpy()


def compute_similarity(embed_t, embed_t_tau, t):
    similarity_scores = torch.mm(embed_t, embed_t_tau.transpose(0, 1)) / t
    return similarity_scores


def cosine_similarity_loss(embed_t, embed_t_tau, DEVICE, t=0.3):
    batch_size = embed_t.shape[0]
    similarity_scores = compute_similarity(embed_t, embed_t_tau, t)
    y = torch.arange(batch_size, device=DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(similarity_scores, y)
    return loss, y, similarity_scores


def accuracy(y_scores, y_true):
    y_pred = y_scores.argmax(dim=-1)
    return (y_pred == y_true).sum().item() / len(y_scores)


def train_step(model, opt, loss_fn, data, idx, device, temp, graph_steps=1):
    model.train()
    for _, idx_batch in enumerate(idx):
        opt.zero_grad()
        total_loss = 0.
        curr_graph_enc = None
        next_graph_enc = None

        for i in range(graph_steps):
            x, adjacency, _ = data.dataset[idx_batch[0] + i]
            curr_graph_enc, curr_embed = model(x, adjacency, curr_graph_enc)
            x, adjacency, _ = data.dataset[idx_batch[0] + i + 1]
            next_graph_enc, next_embed = model(x, adjacency, next_graph_enc)
            loss, y, similarity_scores = loss_fn(curr_embed, next_embed, device, t=temp)
            total_loss = total_loss + loss

        total_loss.backward()
        opt.step()

    with torch.no_grad():
        acc = accuracy(similarity_scores, y)
    return total_loss.item(), acc


def evaluate(model, data, idx, num_labels, graph_steps=1):
    """
    We need to have ground truth labels for computing ARI.
    """
    model.eval()
    with torch.no_grad():
        embeddings = []
        y_true = []
        for _, idx_batch in enumerate(idx):
            graph_enc = None
            for k in range(graph_steps):
                x, adjacency, y = data.dataset[idx_batch[0] + k]
                graph_enc, embed = model(x, adjacency, graph_enc)
            embeddings.append(embed)
            y_true.append(y)
        embeddings = torch.cat(embeddings).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()
        kmeans = KMeans(n_clusters=num_labels)
        y_pred = kmeans.fit_predict(embeddings)
        ari = adjusted_rand_score(y_true, y_pred)

    return embeddings, ari, y_true, y_pred


def create_dict(embedding, idx, labels_emb):
    values = list(zip(labels_emb, embedding))
    keys = idx.tolist()
    dict_graphs = dict(zip(keys, values))
    sorted_keys = dict(sorted(dict_graphs.items()))

    return list(sorted_keys.values())
