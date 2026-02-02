from typing import List
from .config import VerMindConfig


class VLMConfig(VerMindConfig):
    model_type = "vermind-v"

    def __init__(
            self,
            image_special_token: str = '<image>', 
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)
