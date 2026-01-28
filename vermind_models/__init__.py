from .base_module import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
from .config import VerMindConfig, VLMConfig
from .models.modeling_vermind import VerMindBlock, VerMindModel, VerMindForCausalLM
from .models.modeling_vermind_v import VisionProj, VerMindVLM
from .FFN import FeedForward
from .GQA import Attention

# Register model and config with transformers AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

# Register config
AutoConfig.register("vermind", VerMindConfig)
AutoConfig.register("vermind-v", VLMConfig)

# Register model
AutoModelForCausalLM.register(VerMindConfig, VerMindForCausalLM)
AutoModelForCausalLM.register(VLMConfig, VerMindVLM)

__all__ = [
    # Base modules
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    "repeat_kv",
    # Config
    "VerMindConfig",
    "VLMConfig",
    # Model components
    "FeedForward",
    "Attention",
    "VisionProj",
    # Model classes
    "VerMindBlock",
    "VerMindModel",
    "VerMindForCausalLM",
    "VerMindVLM",
]
