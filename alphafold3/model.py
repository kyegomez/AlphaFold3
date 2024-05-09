import torch
from torch import Tensor, nn

from alphafold3.diffusion import GeneticDiffusion
from alphafold3.pairformer import PairFormer

# structure module


class AlphaFold3(nn.Module):
    """
    AlphaFold3 model implementation.

    Args:
        dim (int): Dimension of the model.
        seq_len (int): Length of the sequence.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        attn_dropout (float): Dropout rate for attention layers.
        ff_dropout (float): Dropout rate for feed-forward layers.
        global_column_attn (bool, optional): Whether to use global column attention. Defaults to False.
        pair_former_depth (int, optional): Depth of the PairFormer blocks. Defaults to 48.
        num_diffusion_steps (int, optional): Number of diffusion steps. Defaults to 1000.
        diffusion_depth (int, optional): Depth of the diffusion module. Defaults to 30.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int,
        dim_head: int,
        attn_dropout: float,
        ff_dropout: float,
        global_column_attn: bool = False,
        pair_former_depth: int = 48,
        num_diffusion_steps: int = 1000,
        diffusion_depth: int = 30,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.global_column_attn = global_column_attn

        self.confidence_projection = nn.Linear(dim, 1)

        # Pairformer blocks
        self.pairformer = PairFormer(
            dim=dim,
            seq_len=seq_len,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            global_column_attn=global_column_attn,
            depth=pair_former_depth,
        )

        # Diffusion module
        self.diffuser = GeneticDiffusion(
            channels=dim,
            num_diffusion_steps=1000,
            training=False,
            depth=diffusion_depth,
        )

    def forward(
        self,
        pair_representation: Tensor,
        single_representation: Tensor,
        return_loss: bool = False,
        ground_truth: Tensor = None,
        return_confidence: bool = False,
        return_embeddings: bool = True,
    ) -> Tensor:
        """
        Forward pass of the AlphaFold3 model.

        Args:
            pair_representation (Tensor): Pair representation tensor.
            single_representation (Tensor): Single representation tensor.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.
            ground_truth (Tensor, optional): Ground truth tensor. Defaults to None.
            return_confidence (bool, optional): Whether to return the confidence. Defaults to False.
            return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

        Returns:
            Tensor: Output tensor based on the specified return type.
        """
        # Recycle bins
        # recyle_bins = []

        # TODO: Input
        # TODO: Template
        # TODO: MSA

        b, n, n_two, dim = pair_representation.shape
        b_two, n_two, dim_two = single_representation.shape

        # Concat
        x = torch.cat(
            [pair_representation, single_representation], dim=1
        )

        # Apply the 48 blocks of PairFormer
        x = self.pairformer(x)
        print(x.shape)

        # Add the embeddings to the recycle bins
        # recyle_bins.append(x)

        # Diffusion
        x = self.diffuser(x, ground_truth)

        # If return_confidence is True, return the confidence
        if return_confidence is True:
            x = self.confidence_projection(x)
            return x

        # If return_loss is True, return the loss
        if return_embeddings is True:
            return x
