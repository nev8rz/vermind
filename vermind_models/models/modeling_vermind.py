from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional,List,Tuple,Union
from transformers import PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


from ..base_module import RMSNorm,precompute_freqs_cis
from ..FFN import FeedForward
from ..config import VerMindConfig
from ..GQA import Attention

class VerMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: VerMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None, position_ids=None, cu_seqlens=None):
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
    def __init__(self, config: VerMindConfig):
        super().__init__()
        self.config = config 
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([VerMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                position_ids: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                **kwargs):
        is_varlen = (cu_seqlens is not None and 
                     input_ids.dim() == 1)
        
        if is_varlen:
            total_tokens = input_ids.shape[0]
            hidden_states = self.dropout(self.embed_tokens(input_ids))
        else:
            batch_size, seq_length = input_ids.shape
            hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) 
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        position_embeddings = (
            self.freqs_cos,
            self.freqs_sin
        )

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
    


class VerMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VerMindConfig

    def __init__(self, config: VerMindConfig = None):
        self.config = config or VerMindConfig()
        super().__init__(self.config)
        self.model = VerMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                position_ids: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            **args
        )
        
        is_varlen = cu_seqlens is not None
        if is_varlen:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if is_varlen:
                shift_logits = logits[:-1, :].contiguous()  # (total_tokens-1, vocab_size)
                shift_labels = labels[1:].contiguous()  # (total_tokens-1,)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
