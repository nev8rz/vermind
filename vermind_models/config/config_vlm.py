from typing import List
from .config import VerMindConfig


class VLMConfig(VerMindConfig):
    model_type = "vermind-v"  # 修改模型类型名称

    def __init__(
            self,
            image_special_token: str = '<image>', 
            # 关键修改：因为使用了 2x2 Pooling，196 个 patch 变成了 49 个
            # 所以占位符 image_ids 的长度必须是 49
            image_ids: List = [34] * 49, 
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)
