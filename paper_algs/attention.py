# Adapted from https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Config:
    def __init__(self, enable_flash, enable_math, enable_mem_efficient):
        self.enable_flash = enable_flash
        self.enable_math = enable_math
        self.enable_mem_efficient = enable_mem_efficient

def default(val, d):
    return val if val is not None else d

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0., gate_output=True, query_bias=True, flash=True, window_size=None, efficient_attn_config=Config(True, True, True)):
        super().__init__()
        dim_inner = dim_head * heads

        self.attend = Attend(flash=flash, dropout=dropout, window_size=window_size, attn_config=efficient_attn_config)

        self.to_q = nn.Linear(dim, dim_inner, bias=query_bias)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

        self.to_gates = None
        if gate_output:
            gate_linear = nn.Linear(dim, dim_inner)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, 1.)
            self.to_gates = gate_linear

    def forward(self, seq, mask=None, context=None, attn_bias=None):
        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.to_q.out_features // self.to_q.in_features), (q, k, v))

        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask)

        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.to_gates is not None:
            gates = self.to_gates(seq)
            out = out * gates.sigmoid()

        return self.to_out(out)


class Attend(nn.Module):
    def __init__(self, dropout=0., flash=False, window_size=None, scale=None, attn_config=Config(True, True, True)):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.is_local_attn = window_size is not None
        self.window_size = window_size
        self.flash = flash
        self.attn_config = attn_config
        self.attn_dropout = nn.Dropout(dropout)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, seq_len, _ = q.shape
        attn_mask = None

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, heads, seq_len, -1)

        with torch.backends.cuda.sdp_kernel(**self.attn_config.__dict__):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale, dropout_p=self.dropout if self.training else 0.)

        return out

    def local_attn(self, q, k, v, mask=None, attn_bias=None):
        window_size, batch, seq_len, device = self.window_size, q.shape[0], q.shape[-2], q.device

        if mask is None:
            mask = torch.ones((batch, seq_len), device=device, dtype=torch.bool)

        padding_needed = (window_size - (seq_len % window_size)) % window_size

        if padding_needed > 0:
            q, k, v = map(lambda t: F.pad(t, (0, 0, 0, padding_needed), value=0.), (q, k, v))
            mask = F.pad(mask, (0, padding_needed), value=False)

        q, k, v = map(lambda t: rearrange(t, 'b h (n w) d -> b h n w d', w=window_size), (q, k, v))
        mask = rearrange(mask, 'b (n w) -> b n w', w=window_size)

        k, v = map(lambda t: F.pad(t, (0, 0, 1, 1)), (k, v))
        mask = F.pad(mask, (1, 1), value=False)

        k, v = map(lambda t: torch.cat((t[..., :-2, :], t[..., 1:-1, :], t[..., 2:, :]), dim=-2), (k, v))
        mask = torch.cat((mask[..., :-2], mask[..., 1:-1], mask[..., 2:]), dim=-1)

        if attn_bias is not None:
            attn_bias = F.pad(attn_bias, (0, padding_needed, 0, padding_needed), value=0.)
            attn_bias = rearrange(attn_bias, '... (i w1) (j w2) -> ... i j w1 w2', w1=window_size, w2=window_size)
            attn_bias = F.pad(attn_bias, (0, 0, 0, 0, 1, 1), value=0.)

            attn_bias = torch.cat((attn_bias[..., :-2, :, :], attn_bias[..., 1:-1, :, :], attn_bias[..., 2:, :, :]), dim=-1)

            merged_batch = attn_bias.shape[0]
            diag_mask = torch.eye(attn_bias.shape[1], device=device, dtype=torch.bool)
            diag_mask = diag_mask.unsqueeze(0).expand(merged_batch, -1, -1)

            attn_bias = attn_bias[diag_mask].view(merged_batch, -1, attn_bias.shape[-2], attn_bias.shape[-1])

        scale = q.shape[-1] ** -0.5
        q = q * scale

        sim = torch.einsum('... i d, ... j d -> ... i j', q, k)

        if attn_bias is not None:
            if attn_bias.ndim == 4:
                attn_bias = attn_bias.unsqueeze(1)
            assert attn_bias.ndim == sim.ndim
            sim = sim + attn_bias

        mask = mask.unsqueeze(1).unsqueeze(3)
        sim = sim.masked_fill(~mask, max_neg_value(sim))

        attn = sim.softmax(dim=-1)

        out = torch.einsum('... i j, ... j d -> ... i d', attn, v)

        out = rearrange(out, 'b h n w d -> b h (n w) d')
        out = out[..., :seq_len, :]

        return out

    def forward(self, q, k, v, mask=None, attn_bias=None):
        if self.is_local_attn:
            return self.local_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        can_use_flash = self.flash and attn_bias is None

        if can_use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        scale = default(self.scale, q.shape[-1] ** -0.5)
        q = q * scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        if attn_bias is not None:
            sim = sim + attn_bias

        if mask is not None:
            mask_value = max_neg_value(sim)
            mask = mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        return out
