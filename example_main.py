import os
import tqdm
import copy
import glob
import argparse
import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.transformer import Transformer
from src.layers_model import PositionalEncoding, ContrastiveProj
from src.utils import cosine_similarity_loss, split_train_test, train_step, evaluate

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str,
                    default='2WellGraph_085')
parser.add_argument('--type_of_data', type=str,
                    default='data_no_positions', help='data_no_positions or data_positions')
parser.add_argument('--data_dir', type=str,
                    default=r"data")
parser.add_argument('--output_dir', type=str,
                    default=r"result")
parser.add_argument('--type_explain', type=str,
                    default='edges')
parser.add_argument('-master_node_dim', type=int, default=32)
parser.add_argument('--h', type=int, default=3, help='Hyperparameter that controls the representation history')
parser.add_argument('--final_proj_dim', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--train_steps', type=int, default=200)
parser.add_argument('--print_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

print(f'Uploading {args.data_name}...')
input_dir = os.path.join(args.data_dir, args.type_of_data, args.data_name)

if args.type_of_data == 'data_positions':
    graphs = np.load(os.path.join(input_dir, 'graphs.npy'))
    position_encoding = PositionalEncoding(args.master_node_dim)

else:
    files = sorted(glob.glob(os.path.join(input_dir, 'graphs', '*')),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    adjacency = []
    for file in tqdm.tqdm(files):
        graph = np.load(file)
        adjacency.append(graph)
    graphs = np.array(adjacency)
    position_encoding = None

labels = np.load(os.path.join(input_dir, 'labels.npy'))
num_labels = len(np.unique(labels))
num_graphs, num_nodes, _ = graphs.shape

adjacency = torch.from_numpy(graphs).to(DEVICE)
x = adjacency.sum(dim=-1).long()
y = torch.from_numpy(labels).long().to(DEVICE)

ds = TensorDataset(x, adjacency, y)
ds = DataLoader(ds)

ari_train_list = []
ari_test_list = []
ari_test = 0
ari_train = 0

# Define model
main_model = Transformer(num_layers=args.num_layers,
                         num_heads=args.num_heads,
                         master_node_dim=args.master_node_dim,
                         ffn_dim=4*args.master_node_dim,
                         num_nodes=num_nodes,
                         device=DEVICE,
                         position_encoding=position_encoding,
                         proj_dim=args.final_proj_dim)

contrastive_model = ContrastiveProj(main_model, args.final_proj_dim, args.master_node_dim)

proj_model = main_model.to(DEVICE)
contrastive_model = contrastive_model.to(DEVICE)

loss_fn = cosine_similarity_loss
optimizer = optim.Adam(contrastive_model.parameters())

running_loss = 0.
running_acc = 0.
best_acc = 0.
projs_train = []
y_preds_train = []
y_trues_train = []

# Splitting the set of indices into train and test sets
train_idx, test_idx, train_idx_to_save, test_idx_to_save = split_train_test(num_graphs,
                                                                            args.batch_size,
                                                                            args.h)
if not os.path.exists(os.path.join(input_dir, args.output_dir)):
    os.mkdir(os.path.join(input_dir, args.output_dir))

print('Start training...')
best_test_ari = 0
best_proj_model = None
best_contrastive_model = None

for i in range(args.train_steps):
    train_loss, train_acc = train_step(model=contrastive_model,
                                       opt=optimizer,
                                       loss_fn=loss_fn,
                                       data=ds,
                                       idx=train_idx,
                                       device=DEVICE,
                                       temp=1.0,
                                       graph_steps=args.h)
    running_loss += train_loss
    running_acc += train_acc
    if i % args.print_steps == args.print_steps - 1:
        curr_proj_train, ari_train, y_true_train, y_pred_train = evaluate(model=proj_model,
                                                                          data=ds,
                                                                          idx=train_idx,
                                                                          num_labels=num_labels,
                                                                           graph_steps=args.h)

        proj_test, ari_test, y_true_test, y_pred_test = evaluate(model=proj_model,
                                                                 data=ds,
                                                                 idx=test_idx,
                                                                 num_labels=num_labels,
                                                                 graph_steps=args.h)
        if ari_test > best_test_ari:
            best_test_ari = ari_test
            print(f'[{i + 1:4d}] ari_test: {ari_test:.4f}')
            best_proj_model = copy.deepcopy(proj_model.state_dict())
            best_contrastive_model = copy.deepcopy(contrastive_model.state_dict())

        projs_train.append(curr_proj_train)
        y_preds_train.append(y_pred_train)
        y_trues_train.append(y_true_train)

        running_loss /= args.print_steps
        running_acc /= args.print_steps
        print(f'[{i+1:4d}] contrastive_loss: {running_loss:.4f}  classification_acc: {running_acc:6.2%} '
              f'ari_train: {ari_train:.4f}')
        running_loss = 0.
        running_acc = 0.

        plt.figure()

        # If the dimension of the final representation is higher than 2, than apply TSNE to visualize
        # the final representation
        if curr_proj_train.shape[-1] > 2:
            tsne = TSNE(n_components=2)
            components = tsne.fit_transform(curr_proj_train)
            plt.scatter(components[:, 0], components[:, 1], c=y_true_train)
        elif curr_proj_train.shape[-1] == 2:
            plt.scatter(curr_proj_train[:, 0], curr_proj_train[:, 1], c=y_true_train)
        else:
            plt.scatter(range(len(curr_proj_train)), curr_proj_train, c=y_true_train)
        plt.savefig(os.path.join(input_dir, args.output_dir,
                                 f'embedding_true_labels_{i}.png'))

embeddings_train = np.array(projs_train)

# Save models and results
# np.savez_compressed(os.path.join(input_dir, args.output_dir, 'embedding_train'),
#                     emb=embeddings_train, y_preds=y_preds_train, y_true=y_trues_train, train_idx=train_idx_to_save)
# torch.save(best_proj_model, os.path.join(input_dir, args.output_dir, 'proj_model'))
# torch.save(best_contrastive_model, os.path.join(input_dir, args.output_dir,
#                                                 'contrastive_model'))

embedding_test, ari, y_true_test, y_pred_test = evaluate(model=proj_model,
                                                         data=ds,
                                                         idx=test_idx,
                                                         num_labels=num_labels,
                                                         graph_steps=args.h)
print('ARI on the test set: ', ari)
# np.savez_compressed(os.path.join(input_dir, args.output_dir, 'embedding_test'),
#                     emb=embedding_test, y_preds=y_pred_test, y_true=y_true_test, test_idx=test_idx_to_save)

