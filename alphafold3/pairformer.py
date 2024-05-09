from alphafold2_pytorch.utils import *
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from typing import Tuple, Optional
import torch
from alphafold3.model import (
    FeedForward,
    AxialAttention,
    TriangleMultiplicativeModule,
)

# structure module


def default(val, d):
    return val if val is not None else d


def exists(val):
    return val is not None


# PairFormer blocks


class OuterMean(nn.Module):
    def __init__(self, dim, hidden_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, "b m i d -> b m i () d") * rearrange(
            right, "b m j d -> b m () j d"
        )

        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(
                mask, "b m i -> b m i () ()"
            ) * rearrange(mask, "b m j -> b m () j ()")
            outer = outer.masked_fill(~mask, 0.0)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)

        return self.proj_out(outer)


class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout=0.0,
        global_column_attn=False,
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=True,
            col_attn=False,
            accept_edges=True,
        )
        self.triangle_attention_ingoing = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=False,
            col_attn=True,
            accept_edges=True,
            global_query_attn=global_column_attn,
        )
        self.triangle_multiply_outgoing = (
            TriangleMultiplicativeModule(dim=dim, mix="outgoing")
        )
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(
            dim=dim, mix="ingoing"
        )

    def forward(self, x, mask=None, msa_repr=None, msa_mask=None):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask=msa_mask)

        x = self.triangle_multiply_outgoing(x, mask=mask) + x
        x = self.triangle_multiply_ingoing(x, mask=mask) + x
        x = (
            self.triangle_attention_outgoing(x, edges=x, mask=mask)
            + x
        )
        x = self.triangle_attention_ingoing(x, edges=x, mask=mask) + x
        return x


class MsaAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=True,
            col_attn=False,
            accept_edges=True,
        )
        self.col_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=False,
            col_attn=True,
        )

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        x = self.col_attn(x, mask=mask) + x
        return x


# main PairFormer class
class PairFormerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        seq_len: int,
        heads: int,
        dim_head: int,
        attn_dropout: float,
        ff_dropout: float,
        global_column_attn: bool = False,
    ):
        """
        PairFormer Block module.

        Args:
            dim: The input dimension.
            seq_len: The length of the sequence.
            heads: The number of attention heads.
            dim_head: The dimension of each attention head.
            attn_dropout: The dropout rate for attention layers.
            ff_dropout: The dropout rate for feed-forward layers.
            global_column_attn: Whether to use global column attention in pairwise attention block.
        """
        super().__init__()
        self.layer = nn.ModuleList(
            [
                PairwiseAttentionBlock(
                    dim=dim,
                    seq_len=seq_len,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=attn_dropout,
                    global_column_attn=global_column_attn,
                ),
                FeedForward(dim=dim, dropout=ff_dropout),
                MsaAttentionBlock(
                    dim=dim,
                    seq_len=seq_len,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=attn_dropout,
                ),
                FeedForward(dim=dim, dropout=ff_dropout),
            ]
        )

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass of the PairFormer Block.

        Args:
            inputs: A tuple containing the input tensors (x, m, mask, msa_mask).

        Returns:
            A tuple containing the output tensors (x, m, mask, msa_mask).
        """
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # msa attention and transition
        m = msa_attn(m, mask=msa_mask, pairwise_repr=x)
        m = msa_ff(m) + m

        # pairwise attention and transition
        x = attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask


class PairFormer(nn.Module):
    def __init__(self, *, depth, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [PairFormerBlock(**kwargs) for _ in range(depth)]
        )

    def forward(self, x, m, mask=None, msa_mask=None):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m
