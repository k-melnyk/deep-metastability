from src.layers_model import *


class Transformer(nn.Module):
    """
    The model that consists of Transformer and contrastive learning.
     The methods are modified for the explainability (https://github.com/hila-chefer/Transformer-Explainability).
    """
    def __init__(self, num_layers, num_heads, master_node_dim, ffn_dim, num_nodes, device,
                 position_encoding=None, proj_dim=None):
        super().__init__()
        self.master_node_dim = master_node_dim
        self.device = device
        self.encoder = Encoder(num_layers, num_heads, master_node_dim, ffn_dim)
        self.proj = Linear(master_node_dim, proj_dim) if proj_dim else None
        self.graph_node = nn.Parameter(torch.randn(master_node_dim), requires_grad=True)
        self.input_embed = nn.Embedding(num_nodes, master_node_dim)
        self.position_encoding = position_encoding
        self.inp_grad = None

    def prepare_input(self, x, adjacency, transformer_out=None):
        batch_size, num_graphs, _ = x.size()
        if transformer_out is None:
            master_node = self.graph_node.repeat(batch_size, 1, 1)
        else:
            master_node = transformer_out.unsqueeze(1)
        x = torch.cat([master_node, x], dim=1)
        x *= math.sqrt(self.master_node_dim)

        updated_adjacency = torch.empty(
            batch_size, num_graphs + 1, num_graphs + 1, device=self.device)
        updated_adjacency[:, 1:, 1:] = adjacency
        updated_adjacency[:, :, 0] = 0
        updated_adjacency[:, 0, :] = 1

        if self.position_encoding is not None:
            x = self.position_encoding(x)
        return x, updated_adjacency

    def forward(self, x, adjacency, tranformer_out=None):
        x = self.input_embed(x)
        x, adjacency = self.prepare_input(x, adjacency, tranformer_out)

        if x.requires_grad:
            x.register_hook(self.save_inp_grad)

        x = self.encoder(x, adjacency)

        master_node = x[:, 0]
        if self.proj is not None:
            final_proj = self.proj(master_node)
            return master_node, final_proj
        else:
            return master_node

    def relprop(self, cam, method=None, **kwargs):
        cam = self.proj.relprop(cam, **kwargs)
        cam = self.encoder.relprop(cam, method, **kwargs)
        return cam

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def compute_bilrp(self, ds, idx1, idx2=None, method=None):
        x, adjacency, _ = ds.dataset[idx1]
        r1 = self.compute_branch(x, adjacency)

        if idx2 is not None:
            x, adjacency, _ = ds.dataset[idx2]
            r2 = self.compute_branch(x, adjacency, method)
            R = [np.asarray(r1).squeeze(), np.asarray(r2).squeeze()]
            R = np.tensordot(R[0], R[1])
            return R

        R = np.dot(r1[0].T, r1[1])

        return R

    def compute_branch(self, x, adjacency, method=None):
        kwargs = {"alpha": 1}
        master_node = None
        master_node, embed = self.forward(torch.unsqueeze(x, 0), adjacency, master_node)
        f = embed.squeeze()

        R = []
        for i, f_i in enumerate(f):
            z = np.zeros(len(f))
            z[i] = f_i
            f_i.backward(retain_graph=True)
            r_proj = torch.FloatTensor(z).to(self.device)
            r = self.relprop(r_proj, method='grad', **kwargs).detach().cpu().numpy()

            R.append(r)
        return R

    def compute_lrp(self, ds, idx1):
        kwargs = {"alpha": 1}
        master_node = None
        x, adjacency, y = ds.dataset[idx1]
        master_node, embed = self.forward(torch.unsqueeze(x, 0), adjacency, master_node)

        self.zero_grad()
        embed.sum().backward(retain_graph=True)
        R = self.relprop(embed.to(self.device), method='grad', **kwargs)

        return R

