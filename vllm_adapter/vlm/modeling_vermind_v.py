# coding=utf-8
"""
Model file for VerMind-V (VLM) model - Standalone Version
Contains complete VLM implementation without external dependencies
"""

import math
import os
import warnings
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_vermind_v import VLMConfig

warnings.filterwarnings('ignore')


# ==================== Base Module Functions ====================

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """Precompute rotary position embedding frequencies"""
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings to queries and keys"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 保存原始 dtype
    orig_dtype = q.dtype

    if position_ids is not None:
        if position_ids.dim() == 1:
            pos_ids = position_ids
            cos_selected = cos[pos_ids]
            sin_selected = sin[pos_ids]
            cos_selected = cos_selected.unsqueeze(0).unsqueeze(2)
            sin_selected = sin_selected.unsqueeze(0).unsqueeze(2)
        else:
            cos_selected = cos[position_ids]
            sin_selected = sin[position_ids]
            cos_selected = cos_selected.unsqueeze(2)
            sin_selected = sin_selected.unsqueeze(2)

        q_embed = (q * cos_selected) + (rotate_half(q) * sin_selected)
        k_embed = (k * cos_selected) + (rotate_half(k) * sin_selected)
    else:
        seq_len = q.shape[1]
        cos_s = cos[:seq_len]
        sin_s = sin[:seq_len]
        cos_s = cos_s.unsqueeze(0).unsqueeze(2)
        sin_s = sin_s.unsqueeze(0).unsqueeze(2)
        q_embed = (q * cos_s) + (rotate_half(q) * sin_s)
        k_embed = (k * cos_s) + (rotate_half(k) * sin_s)

    # 转回原始 dtype
    q_embed = q_embed.to(orig_dtype)
    k_embed = k_embed.to(orig_dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
        bs, slen, num_key_value_heads * n_rep, head_dim
    )


# ==================== Module Classes ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class Attention(nn.Module):
    """Grouped Query Attention with RoPE"""
    def __init__(self, args):
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

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, position_ids=None, cu_seqlens=None):
        bsz, seq_len, _ = x.shape
        # 获取权重的 dtype（模型加载时的 dtype）
        weight_dtype = self.q_proj.weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(weight_dtype)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 强制统一为权重 dtype（防止不同 proj 层 dtype 不一致）
        xq = xq.to(weight_dtype)
        xk = xk.to(weight_dtype)
        xv = xv.to(weight_dtype)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, position_ids=position_ids)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2)

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


class VerMindBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, position_ids=None, cu_seqlens=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class VerMindModel(nn.Module):
    """VerMind Model (Transformer backbone)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([VerMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def _cast_buffers_to_dtype(self, dtype):
        """将 position embedding buffers 转换为指定 dtype（跟随模型权重）"""
        if self.freqs_cos.dtype != dtype:
            self.freqs_cos = self.freqs_cos.to(dtype)
            self.freqs_sin = self.freqs_sin.to(dtype)
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, position_ids=None, cu_seqlens=None, **kwargs):
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 确保 position embeddings 与 hidden_states 同 dtype
        self._cast_buffers_to_dtype(hidden_states.dtype)
        
        position_embeddings = (self.freqs_cos, self.freqs_sin)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, position_ids=None, cu_seqlens=None, **kwargs):
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos, self.freqs_sin)

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = 0
        return hidden_states, presents, aux_loss


class VisionProj(nn.Module):
    """Vision Projection Layer - Projects vision features to language model space"""
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        intermediate_size = min(ve_hidden_size, hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(ve_hidden_size),
            nn.Linear(ve_hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, image_encoders):
        return self.proj(image_encoders)


class VerMindVLM(PreTrainedModel, GenerationMixin):
    """VerMind Vision-Language Model"""
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="google/siglip-base-patch16-224"):
        self.params = params or VLMConfig()
        # Call PreTrainedModel init directly
        super().__init__(self.params)
        
        # Build the model components
        self.model = VerMindModel(self.params)
        self.lm_head = nn.Linear(self.params.hidden_size, self.params.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # Vision components
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(ve_hidden_size=768, hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        """Load vision encoder (SigLIP)"""
        from transformers import logging as hf_logging
        from transformers import SiglipVisionModel, SiglipProcessor
        hf_logging.set_verbosity_error()
        
        if not os.path.exists(model_path) and "/" not in model_path:
            return None, None
        
        print(f"[VerMind-V] Loading Vision Encoder: {model_path}...")
        try:
            vision_model = SiglipVisionModel.from_pretrained(model_path)
            processor = SiglipProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading SigLIP Vision: {e}")
            return None, None

        for param in vision_model.parameters():
            param.requires_grad = False
        return vision_model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """Convert PIL image to tensor"""
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        """Extract image features from vision encoder"""
        outputs = vision_model(pixel_values=image_tensors)
        return outputs.last_hidden_state

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """Insert vision projections into hidden states at image token positions"""
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                           matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            # 确保 vision_proj 与 hidden_states 数据类型一致
            vision_proj = vision_proj.to(h.dtype)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if vision_proj.dim() == 4:
                            current_vision_embeds = vision_proj[0, i]
                        else:
                            current_vision_embeds = vision_proj[i]
                        
                        if img_idx < 1:
                            h_i = torch.cat((h_i[:start_idx], current_vision_embeds, h_i[end_idx + 1:]), dim=0)[:seqlen]
                        img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, logits_to_keep=0,
                pixel_values=None, **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 5:
                pixel_values = pixel_values[:, 0, :, :, :]
            vision_tensors = VerMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1]
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output


class VerMindVLMModel(VerMindModel):
    """VerMind-V Model (extends VerMindModel for VLM)"""
    pass  # Inherits everything from VerMindModel


# Register the model class
AutoModelForCausalLM.register(VerMindVLM.config_class, VerMindVLM)

__all__ = ["VerMindVLM", "VisionProj", "VerMindVLMModel"]
