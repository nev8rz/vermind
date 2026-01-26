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

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN_VARLEN = True
except ImportError:
    HAS_FLASH_ATTN_VARLEN = False

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
                position_ids: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None):
        use_varlen = (cu_seqlens is not None and 
                      HAS_FLASH_ATTN_VARLEN and 
                      self.flash and 
                      past_key_value is None and
                      x.dim() == 2)  # varlen 模式下 x 应该是 (total_tokens, hidden_size)，不是 (batch, seq, hidden)
        
        if use_varlen:
            total_tokens, hidden_size = x.shape
            if x.dtype == torch.float32:
                x = x.to(torch.bfloat16)
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            if xq.dtype == torch.float32:
                xq = xq.to(torch.bfloat16)
                xk = xk.to(torch.bfloat16)
                xv = xv.to(torch.bfloat16)
            xq = xq.view(total_tokens, self.n_local_heads, self.head_dim)
            xk = xk.view(total_tokens, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(total_tokens, self.n_local_kv_heads, self.head_dim)
            
            cos, sin = position_embeddings
            if position_ids is not None:
                xq_reshaped = xq.unsqueeze(0)
                xk_reshaped = xk.unsqueeze(0)
                xq_emb, xk_emb = apply_rotary_pos_emb(xq_reshaped, xk_reshaped, cos, sin, position_ids=position_ids)
                xq, xk = xq_emb.squeeze(0), xk_emb.squeeze(0)
            else:
                xq, xk = apply_rotary_pos_emb(xq.unsqueeze(0), xk.unsqueeze(0), cos, sin)
                xq, xk = xq.squeeze(0), xk.squeeze(0)
            
            xk = repeat_kv(xk.unsqueeze(0), self.n_rep).squeeze(0)
            xv = repeat_kv(xv.unsqueeze(0), self.n_rep).squeeze(0)
            
            if xq.dtype not in (torch.float16, torch.bfloat16):
                xq = xq.to(torch.bfloat16)
            if xk.dtype not in (torch.float16, torch.bfloat16):
                xk = xk.to(torch.bfloat16)
            if xv.dtype not in (torch.float16, torch.bfloat16):
                xv = xv.to(torch.bfloat16)
            
            batch_size = cu_seqlens.shape[0] - 1
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            
            output = flash_attn_varlen_func(
                xq, xk, xv,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
            
            output = output.reshape(total_tokens, -1)
            output = self.resid_dropout(self.o_proj(output))
            return output, None
        
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
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

        is_2d_mask = attention_mask is not None and attention_mask.dim() == 3
        attn_mask_for_flash = None
        use_flash = False
        
        if self.flash and (seq_len > 1) and (past_key_value is None):
            if attention_mask is None:
                use_flash = True
                attn_mask_for_flash = None
            elif is_2d_mask:
                use_flash = False
            elif torch.all(attention_mask == 1):
                use_flash = True
                attn_mask_for_flash = None
            else:
                use_flash = False
        
        if use_flash:
            if attn_mask_for_flash is not None:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=attn_mask_for_flash,
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=False
                )
            else:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=True
                )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if not is_2d_mask:
                scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                if is_2d_mask:
                    attention_mask = attention_mask[:, 0, :] if attention_mask.dim() == 3 else attention_mask
                
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask.float()) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv