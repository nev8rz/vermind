
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    bias: bool = True