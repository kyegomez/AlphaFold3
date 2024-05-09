import torch 
from alphafold3 import AlphaFold3

x = torch.randn(1, 5, 5, 64)
y = torch.randn(1, 5, 64)

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
output = model(x, y)
print(output.shape)
