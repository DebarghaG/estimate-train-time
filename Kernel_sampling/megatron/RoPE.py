import torch
import math


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=precision) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached
    

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):  # jitting fails with bf16
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)



class RoPE_CLASS(torch.nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype


    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in earlier torch versions


    @torch.jit.script
    def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


    def apply_rotary_pos_emb_torch(
        q, k, cos, sin, offset: int = 0
    ):  # jitting fails with bf16
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)



    def forward(self, query_layer, key_layer, value_layer):
        
        rotary_emb = RotaryEmbedding(self.dim, base=10000, precision=self.dtype)


        query_rot, key_rot = query_layer, key_layer

        
        apply_rotary_fn = (apply_rotary_pos_emb_torch if self.dtype == torch.bfloat16 else apply_rotary_pos_emb)

        seq_len = key_layer.shape[0]
        offset = 0
        # if exists(layer_past) and layer_past.numel() > 0:
        #     offset = layer_past[0].shape[0]
        #     seq_len += offset

        cos, sin = rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = apply_rotary_fn(
            query_rot, key_rot, cos, sin, offset=offset
        )

        return query_layer, key_layer