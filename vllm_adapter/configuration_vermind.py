# coding=utf-8
# Copyright 2024 VerMind Team
# Configuration file for VerMind model

import sys
import os

# Add the vermind_models package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Import and register the config with Transformers
from transformers import AutoConfig
from vermind_models.config import VerMindConfig

# Register the config class
AutoConfig.register("vermind", VerMindConfig)

# Export the config class
__all__ = ["VerMindConfig"]
