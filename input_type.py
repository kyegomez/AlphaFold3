import torch

batch_size = 1
num_nodes = 5
num_features = 64

pair_representations = torch.randn(
    batch_size, num_nodes, num_nodes, num_features
)
single_representations = torch.randn(
    batch_size, num_nodes, num_features
)
