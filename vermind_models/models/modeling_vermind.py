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

class VerMindBlock(nn.Module): # decoder
    def __init__(self, layer_id: int, config: VerMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # 注意力头数
        self.hidden_size = config.hidden_size # 隐藏层输入维度，一般也是dmodel
        self.head_dim = config.hidden_size // config.num_attention_heads # 每个头分到的维度
        self.self_attn = Attention(config) # 注意力

        self.layer_id = layer_id # block id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 输入rmsnorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # # 输出归一化
        # self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.mlp = FeedForward(config) # swiglu 

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 原始值
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), # prenorm
            position_embeddings,
            past_key_value, 
            use_cache, 
            attention_mask
        ) # 
        hidden_states += residual # 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # prenorm -> swiglu -> 残差连接
        return hidden_states, present_key_value # 隐藏层输入，当前kv cache



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
                **kwargs):
        batch_size, seq_length = input_ids.shape # inputs [B,T]
        if hasattr(past_key_values, 'layers'): past_key_values = None # 兼容，可能有的框架是用的kv cache2，这里退化了，不走kv cache了
        past_key_values = past_key_values or [None] * len(self.layers) 
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 已经缓存了多少个历史 token 的 KV，
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        ) # position embeddings

        presents = [] # kv cache
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            ) # hidden states, kv cache
            presents.append(present) # kv cache

        hidden_states = self.norm(hidden_states) # 输出归一化

        # aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
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
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        ) # 模型输出，hidden states, kv cache, aux loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # slice_indices 是用于切片，只保留最后 logits_to_keep 个 token 的 logits
        # lm_head 是用于将 hidden states 映射到 logits[B,T,V],同样只保留最后 logits_to_keep 个 token 的 logits

        # 计算损失
        loss = None
        if labels is not None:
            # 在这里进行shift，将 logits 和 labels 都向右移动一位，然后计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100) # 交叉熵损失

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states) # 格式化输出
        output.aux_loss = aux_loss
        return output