# coding=utf-8
# Model file for VerMind-V (VLM) model

import sys
import os

# Add the vermind_models package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Import and register the model with Transformers
from transformers import AutoModelForCausalLM
from vermind_models.models.modeling_vermind_v import VerMindVLM

# Register the model class
AutoModelForCausalLM.register(VerMindVLM.config_class, VerMindVLM)

# Export the model class
__all__ = ["VerMindVLM"]
