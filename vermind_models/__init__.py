from .base_module import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
from .config import VerMindConfig
from .models.modeling_vermind import VerMindBlock, VerMindModel, VerMindForCausalLM
from .FFN import FeedForward
from .GQA import Attention

# Register model and config with transformers AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

# Register config
AutoConfig.register("vermind", VerMindConfig)

# Register model
AutoModelForCausalLM.register(VerMindConfig, VerMindForCausalLM)

__all__ = [
    # Base modules
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    "repeat_kv",
    # Config
    "VerMindConfig",
    # Model components
    "FeedForward",
    "Attention",
    # Model classes
    "VerMindBlock",
    "VerMindModel",
    "VerMindForCausalLM",
]
