from torch import nn
import torch
from typing import Tuple,Optional
import math
import torch.nn.functional as F

from .config import VerMindConfig
from .base_module import (
    apply_rotary_pos_emb,
    repeat_kv,
    
)

class Attention(nn.Module):
    def __init__(self, args: VerMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 如果提供了 position_ids，使用它来正确索引 position embeddings
        # 这对于打包数据很重要：每个样本应该从位置 0 开始
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, position_ids=position_ids)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 检查是否是 2D mask
        is_2d_mask = attention_mask is not None and attention_mask.dim() == 3
        
        # 准备 flash attention 的 attn_mask（如果使用）
        attn_mask_for_flash = None
        use_flash = False
        
        if self.flash and (seq_len > 1) and (past_key_value is None):
            if attention_mask is None:
                # 没有 mask，使用简单的 causal mask
                use_flash = True
                attn_mask_for_flash = None
            elif is_2d_mask:
                # 2D mask: (batch_size, seq_len, seq_len)
                # 转换为 flash attention 需要的格式：0 可 attend，大负值不可 attend
                # 使用 -1e9 而非 -inf：padding 行全 0 会变成全 mask，-inf 导致 softmax 出 NaN
                attn_mask_for_flash = attention_mask.float()  # (batch_size, seq_len, seq_len)
                attn_mask_for_flash = (1.0 - attn_mask_for_flash) * -1e9
                attn_mask_for_flash = attn_mask_for_flash.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
                use_flash = True
            elif torch.all(attention_mask == 1):
                # 所有位置都是 1，等同于没有 mask，使用 causal
                use_flash = True
                attn_mask_for_flash = None
            else:
                # 1D mask，flash attention 不支持，回退到手动计算
                use_flash = False
        
        if use_flash:
            # 使用 flash attention
            if attn_mask_for_flash is not None:
                # 使用自定义 2D mask，不能同时使用 is_causal
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=attn_mask_for_flash,
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=False  # 使用 attn_mask 时不能使用 is_causal
                )
            else:
                # 使用默认的 causal mask
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=True
                )
        else:
            # 手动计算 attention（支持所有 mask 类型）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 只有在没有提供 2D mask 时才添加默认的 causal mask
            # 如果提供了 2D mask，它已经包含了 causal mask 和样本隔离
            if not is_2d_mask:
                scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                # 支持 2D attention mask (batch_size, seq_len, seq_len) 或 1D mask (batch_size, seq_len)
                if is_2d_mask:
                    # 2D mask: (batch_size, seq_len, seq_len)
                    # 需要扩展到 (batch_size, n_heads, seq_len, seq_len)
                    extended_attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
                    extended_attention_mask = (1.0 - extended_attention_mask.float()) * -1e9
                    scores = scores + extended_attention_mask
                else:
                    # 1D mask: (batch_size, seq_len) - 兼容旧代码
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = (1.0 - extended_attention_mask.float()) * -1e9
                    scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv