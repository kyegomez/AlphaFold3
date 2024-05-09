import torch
from alphafold3.diffusion import GeneticDiffusion

# Create an instance of the GeneticDiffusionModuleBlock
model = GeneticDiffusion(channels=3, training=True)

# Generate random input coordinates
input_coords = torch.randn(10, 100, 100, 3)

# Generate random ground truth coordinates
ground_truth = torch.randn(10, 100, 100, 3)

# Pass the input coordinates and ground truth coordinates through the model
output_coords, loss = model(input_coords, ground_truth)

# Print the output coordinates
print(output_coords)

# Print the loss value
print(loss)
