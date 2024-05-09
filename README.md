[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# AlphaFold3
Implementation of Alpha Fold 3 from the paper: "Accurate structure prediction of biomolecular interactions with AlphaFold3" in PyTorch


## install
`$pip install alphafold3`

## Input Tensor Size Example

```python
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
```

## Genetic Diffusion
Need review but basically it operates on atomic coordinates.

```python
import torch
from alphafold3.diffusion import GeneticDiffusionModuleBlock

# Create an instance of the GeneticDiffusionModuleBlock
model = GeneticDiffusionModuleBlock(channels=3, training=True)

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
```

## Full Model Example Forward pass

```python
import torch 
from alphafold3 import AlphaFold3

# Create random tensors
x = torch.randn(1, 5, 5, 64)  # Shape: (batch_size, seq_len, seq_len, dim)
y = torch.randn(1, 5, 64)  # Shape: (batch_size, seq_len, dim)

# Initialize AlphaFold3 model
model = AlphaFold3(
    dim=64,
    seq_len=5,
    heads=8,
    dim_head=64,
    attn_dropout=0.0,
    ff_dropout=0.0,
    global_column_attn=False,
    pair_former_depth=48,
    num_diffusion_steps=1000,
    diffusion_depth=30,
)

# Forward pass through the model
output = model(x, y)

# Print the shape of the output tensor
print(output.shape)
```


# Citation
```bibtex
@article{Abramson2024-fj,
  title    = "Accurate structure prediction of biomolecular interactions with
              {AlphaFold} 3",
  author   = "Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans,
              Richard and Green, Tim and Pritzel, Alexander and Ronneberger,
              Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick,
              Joshua and Bodenstein, Sebastian W and Evans, David A and Hung,
              Chia-Chun and O'Neill, Michael and Reiman, David and
              Tunyasuvunakool, Kathryn and Wu, Zachary and {\v Z}emgulyt{\.e},
              Akvil{\.e} and Arvaniti, Eirini and Beattie, Charles and
              Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and
              Congreve, Miles and Cowen-Rivers, Alexander I and Cowie, Andrew
              and Figurnov, Michael and Fuchs, Fabian B and Gladman, Hannah and
              Jain, Rishub and Khan, Yousuf A and Low, Caroline M R and Perlin,
              Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and
              Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine
              and Yakneen, Sergei and Zhong, Ellen D and Zielinski, Michal and
              {\v Z}{\'\i}dek, Augustin and Bapst, Victor and Kohli, Pushmeet
              and Jaderberg, Max and Hassabis, Demis and Jumper, John M",
  journal  = "Nature",
  month    =  may,
  year     =  2024
}
```



sequences, ligands, ,covalent bonds -> input embedder [3] -> 


# Todo

- [ ] Implement Figure A, implement triangle update, transition, 
- [ ] Impelment Figure B, per token, cond, 
- [ ] Implement Figure C: Network Chunk,
- [ ] Implement confidence module
- [ ] Implement Template Module
