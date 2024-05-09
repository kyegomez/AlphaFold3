import torch

# Define the batch size, number of nodes, and number of features
batch_size = 1
num_nodes = 5
num_features = 64

# Generate random pair representations using torch.randn
# Shape: (batch_size, num_nodes, num_nodes, num_features)
pair_representations = torch.randn(
    batch_size, num_nodes, num_nodes, num_features
)

# Generate random single representations using torch.randn
# Shape: (batch_size, num_nodes, num_features)
single_representations = torch.randn(
    batch_size, num_nodes, num_features
)
