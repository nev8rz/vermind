# coding=utf-8
"""
Model file for VerMind-V (VLM) model - Standalone Version
Contains complete VLM implementation without external dependencies
"""

import os
import warnings
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..core.modeling_vermind import VerMindForCausalLM, VerMindModel, RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
from .configuration_vermind_v import VLMConfig

warnings.filterwarnings('ignore')


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


class VerMindVLM(VerMindForCausalLM):
    """VerMind Vision-Language Model"""
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="google/siglip-base-patch16-224"):
        # Initialize using parent init but with our own model structure
        self.params = params or VLMConfig()
        # Call PreTrainedModel init directly to avoid double initialization
        nn.Module.__init__(self)
        self.config = self.params
        
        # Build the model components
        self.model = VerMindVLMModel(self.params)
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
