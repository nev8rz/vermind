# coding=utf-8
# Copyright 2024 VerMind Team
# Model file for VerMind model

import sys
import os

# Add the vermind_models package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Import and register the model with Transformers
from transformers import AutoModelForCausalLM
from vermind_models.models.modeling_vermind import VerMindForCausalLM

# Register the model class
AutoModelForCausalLM.register(VerMindForCausalLM.config_class, VerMindForCausalLM)

# Export the model class
__all__ = ["VerMindForCausalLM"]
