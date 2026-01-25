import math
import torch
from torch import nn
from typing import Optional

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # 在网络训练刚开始时，我们希望 Normalization 层不要对数据分布做剧烈的改变，仅仅起到“标准化”的作用。

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    # 强制将数据转为 FP32 (单精度)。FP32 的数值范围和精度远大于 FP16/BF16，可以安全地进行平方累加运算。


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # $\theta_i = 10000^{-2i/d}$，attn_factor默认是1（不进行外推）
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base)) # # 计算每个频率对应的波长，判断它是“高频”还是“低频”
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1) # # 计算 ramp (斜坡函数)：0 表示高频区，1 表示低频区，中间是过渡区
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # $$\begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    if position_ids is not None:
        # 使用 position_ids 按样本索引 cos/sin。打包数据下每个 batch 的 position_ids 不同，必须按行索引。
        # position_ids: (batch_size, seq_len) 或 (seq_len,)
        # cos, sin: (max_seq_len, head_dim)
        # q, k: (batch_size, seq_len, num_heads, head_dim)
        
        if position_ids.dim() == 1:
            pos_ids = position_ids  # (seq_len,)
            cos_selected = cos[pos_ids]   # (seq_len, head_dim)
            sin_selected = sin[pos_ids]   # (seq_len, head_dim)
            cos_selected = cos_selected.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
            sin_selected = sin_selected.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
        else:
            # (batch_size, seq_len) -> 每个样本用自己的 position_ids 索引
            # cos[position_ids] -> (batch_size, seq_len, head_dim)
            cos_selected = cos[position_ids]  # (B, S, head_dim)
            sin_selected = sin[position_ids]  # (B, S, head_dim)
            cos_selected = cos_selected.unsqueeze(2)  # (B, S, 1, head_dim)
            sin_selected = sin_selected.unsqueeze(2)  # (B, S, 1, head_dim)
        
        q_embed = (q * cos_selected) + (rotate_half(q) * sin_selected)
        k_embed = (k * cos_selected) + (rotate_half(k) * sin_selected)
    else:
        # 无 position_ids：使用绝对位置 0..seq_len-1，从 cos/sin 取前 seq_len 再广播
        # cos, sin: (max_seq_len, head_dim); q, k: (bsz, seq_len, num_heads, head_dim)
        seq_len = q.shape[1]
        cos_s = cos[:seq_len]   # (seq_len, head_dim)
        sin_s = sin[:seq_len]   # (seq_len, head_dim)
        cos_s = cos_s.unsqueeze(0).unsqueeze(2)   # (1, seq_len, 1, head_dim)
        sin_s = sin_s.unsqueeze(0).unsqueeze(2)   # (1, seq_len, 1, head_dim)
        q_embed = (q * cos_s) + (rotate_half(q) * sin_s)
        k_embed = (k * cos_s) + (rotate_half(k) * sin_s)
    
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
