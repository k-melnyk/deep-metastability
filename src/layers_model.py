# The code for the explainability is takes from https://github.com/hila-chefer/Transformer-Explainability

import torch.functional as F

from src.utils import *
from src.layers_lrp import *


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = Linear(d_model, d_ff)
        self.W_2 = Linear(d_ff, d_model)
        self.ReLU = ReLU()

    def forward(self, x):
        x = self.W_1(x)
        x = self.ReLU(x)
        x = self.W_2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.W_2.relprop(cam, **kwargs)
        cam = self.ReLU.relprop(cam, **kwargs)
        cam = self.W_1.relprop(cam, **kwargs)
        return cam


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = Linear(d_model, d_model * 3)
        self.W_o = Linear(d_model, d_model)
        self.softmax = Softmax(dim=-1)

        self.matmul1 = einsum('bhid,bhjd->bhij')
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x):
        batch_size, num_heads, seq_len, dk = x.size()
        d_model = num_heads * dk
        return x.transpose(2, 1).reshape(batch_size, seq_len, d_model)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def save_v(self, v):
        self.v = v

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def save_v_cam(self, cam):
        self.v_cam = cam

    def save_attn(self, attn):
        self.attn = attn

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def get_attn(self):
        return self.attn

    def get_attn_gradients(self):
        return self.attn_gradients

    def get_v_cam(self):
        return self.v_cam

    def forward(self, x, mask=None):
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.d_model, dim=2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        self.save_v(v)

        if mask is not None:
            mask = mask.unsqueeze(1)

        d_k = q.size(-1)
        attn_logits = self.matmul1([q, k]) / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn_logits)

        self.save_attn(attn)
        if attn.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        head_attn = self.matmul2([self.attn, v])
        head_attn = self.concat_heads(head_attn)

        mh_attn = self.W_o(head_attn)
        return mh_attn

    def relprop(self, cam, **kwargs):
        cam = self.W_o.relprop(cam, **kwargs)
        cam = self.split_heads(cam)
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = torch.cat((self.concat_heads(cam_q),
                             self.concat_heads(cam_k),
                             self.concat_heads(cam_v)), dim=2)
        cam = self.qkv.relprop(cam_qkv, **kwargs)

        return cam


class Block(nn.Module):
    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.add1, self.add2 = Add(), Add()
        self.clone1, self.clone2 = Clone(), Clone()

    def forward(self, x, neighborhood_mask=None):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.mha(x2, neighborhood_mask)])
        x = self.norm1(x)

        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.ffn(x2)])
        x = self.norm2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.norm2.relprop(cam, **kwargs)
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.ffn.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        cam = self.norm1.relprop(cam, **kwargs)
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.mha.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, d_ff):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(num_heads, d_model, d_ff)
             for _ in range(num_layers)])

    def forward(self, x, adj=None):
        for block in self.blocks:
            x = block(x, neighborhood_mask=adj)
        return x

    def relprop(self, cam, method=None, **kwargs):
        print("conservation 1", cam.sum())

        for block in reversed(self.blocks):
            cam = block.relprop(cam, **kwargs)

        if method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.mha.get_attn_gradients()
                cam = blk.mha.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=0)
            cam = rollout[:, 0, 1:]
            return cam
        return cam


class MLP(nn.Module):
    def __init__(self, final_embed_dim, model_dim):
        super().__init__()
        self.linear1 = Linear(final_embed_dim, 2 * model_dim)
        self.linear2 = Linear(2 * model_dim, model_dim // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.linear2.relprop(cam, **kwargs)
        cam = self.relu.relprop(cam, **kwargs)
        cam = self.linear1.relprop(cam, **kwargs)
        return cam


class ContrastiveProj(nn.Module):
    def __init__(self, gnn, final_embed_dim, model_dim):
        super().__init__()
        self.gnn = gnn
        self.mlp = MLP(final_embed_dim, model_dim)

    def forward(self, x, adjacency, graph_enc=None):
        graph_enc, embedding = self.gnn(x, adjacency, graph_enc)
        projection = self.mlp(embedding) # The projection head for the contrastive learning
        projection = (projection - projection.mean()) / projection.std()
        return graph_enc, projection

    def relprop(self, cam=None, **kwargs):
        cam = self.mlp.relprop(cam, **kwargs)
        cam = self.gnn.relprop(cam, **kwargs)

        return cam


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        assert d_model % 2 == 0
        pe = utils.positional_encoding(d_model, max_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        num_graphs = x.size(1)
        pe = self.pe[:, :num_graphs]
        return x + pe
