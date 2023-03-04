import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    @staticmethod
    def gradprop(Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if not torch.is_tensor(self.X):
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


class AddEye(RelPropSimple):
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)


class ReLU(nn.ReLU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


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


class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)


class CosineSimilarity(RelPropSimple):
    def forward(self, x_t, x_t_tau):
        return utils.compute_similarity(x_t, x_t_tau)


class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R


# class Add(RelPropSimple):
#     @staticmethod
#     def forward(self, inputs):
#         return torch.add(*inputs)
#
#     def relprop(self, R, alpha):
#         Z = self.forward(self.X)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S)
#
#         a = self.X[0] * C[0]
#         b = self.X[1] * C[1]
#
#         a_sum = a.sum()
#         b_sum = b.sum()
#
#         a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
#         b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
#
#         a = a * safe_divide(a_fact, a.sum())
#         b = b * safe_divide(b_fact, b.sum())
#
#         outputs = [a, b]
#
#         return outputs


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, data, idx, graph_steps, DEVICE, start_layer=0):
        curr_graph_enc = None
        next_graph_enc = None
        for k in range(graph_steps):
            x, adjacency, y = data.dataset[idx[k]]
            curr_graph_enc, curr_embed = self.model(x[None, :], adjacency[None, :, :], curr_graph_enc)

            x, adjacency, y = data.dataset[idx[k] + 1]
            next_graph_enc, next_embed = self.model(x[None, :], adjacency[None, :, :], next_graph_enc)
        kwargs = {"alpha": 1}
        sim_score = utils.compute_similarity(curr_embed.requires_grad_(True),
                                             next_embed.requires_grad_(True))

        one_hot_vector = torch.tensor([[sim_score, 0]])

        self.model.zero_grad()
        sim_score.backward(retain_graph=True)

        rel_score = self.model.relprop(one_hot_vector.to(DEVICE), start_layer=start_layer, **kwargs)

        return rel_score