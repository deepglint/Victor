import os
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from transformers import AutoConfig, AutoProcessor
from safetensors.torch import load_file
from torch import nn
import torch
import warnings
# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2_5_VisionTransformerPretrainedModelForLLaVA(nn.Module):
    def __init__(self, model_path,args):
        super().__init__()

        self.is_loaded = False
        self.model_path = model_path
        self.vision_tower_name = model_path
        self.select_layer = -1
        self.select_feature = 'patch'
        self.min_token=getattr(args,"mm_min_image_token",4)
        self.max_token=getattr(args,"mm_max_image_token",2048)
        self.resize_image_size=getattr(args,"resize_image_size",None)
        self.load_model()

    def load_model(self):
        device_map = "auto"  # 或你想要的 device_map
        print(f"{GREEN}Loading QwenViT ...{RESET}")
        visual_model = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(
            self.model_path, use_flash_attention_2=True
        )
        print(f"{GREEN}QwenViT loaded successfully!{RESET}")
        self.vision_tower = visual_model
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        self.reset_image_processor(self.min_token, self.max_token)
        
    def reset_image_processor(self, min_tokens, max_tokens):
        min_pixels=min_tokens*28*28
        max_pixels=max_tokens*28*28
        self.image_processor = AutoProcessor.from_pretrained(self.model_path,min_pixels=min_pixels,max_pixels=max_pixels)
        self.image_processor.resize_image_size=self.resize_image_size
        # Simplified output format
        print(f"{GREEN}MIN_PIXELS: {min_tokens} * 28 * 28 \nMAX_PIXELS: {max_tokens} * 28 * 28{RESET}")
                
    def forward(self, pixel_values, grid_thw=None):
        """
        pixel_values:[all_seq_len,patch_size*patch_size*3*2]
        image_grid_thw:[num_img,3],每个长度为3的向量为[1,h,w],1表示时间,如果为video,则会大于1.h,w为图像的高和宽(以patch为单位)
        """
        # assert 1==2, f'pixel_values:{[x.shape for x in pixel_values]}, grid_thw:{grid_thw}'
        bs = pixel_values.shape[0] if type(pixel_values) is torch.Tensor else len(pixel_values)
        image_features = []
        
        for i in range(bs):
            if grid_thw==None:
                grid_thw = torch.tensor([[1, 40, 40]] * 1).to(pixel_values.device)
                image_features.append(self.vision_tower(pixel_values[i], grid_thw=grid_thw))  # [all_seq_len//4,hidden_size(1536)]
            else:
                image_features.append(self.vision_tower(pixel_values[i].to(self.device), grid_thw=grid_thw[i].to(self.device)))  # [all_seq_len//4,hidden_size(1536)]

        return  image_features
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):#!应该需要改成bf16
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.out_hidden_size

class Args:
    mm_min_image_token = 4
    mm_max_image_token = 2048
    resize_image_size = None

# if __name__=="__main__":
#     model_path = "/vlm/pretrain_models/Qwen2.5-VL-7B-Instruct"
#     args = Args()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Qwen2_5_VisionTransformerPretrainedModelForLLaVA(model_path, args)
#     model.eval().to(device, dtype=torch.bfloat16)  # Move model to device and set to evaluation mode
#     image_path = '/vlm/yinxie/code/sa_514193.jpg'
#     from PIL import Image
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((560, 560), Image.BICUBIC)  # Resize to 256x256
#     image_tensor = model.image_processor.image_processor(image, return_tensors="pt")["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)  # Add batch dimension and move to device
#     print('image.shape', image_tensor.shape)  # Should be (1, 256, 256, 3)

#     from safetensors.torch import save_file
#     import json

#     # 保存权重
#     state_dict = model.vision_tower.state_dict()
#     save_file(state_dict, "/vlm/pretrain_models/Qwen-vit/qwen2_5_vit-7b-patch14-native/model.safetensors")
#     print("vision_tower.safetensors 已保存")

#     # 保存config
#     vision_config = model.vision_tower.config
#     with open("/vlm/pretrain_models/Qwen-vit/qwen2_5_vit-7b-patch14-native/config.json", "w") as f:
#         json.dump(vision_config.to_dict(), f, indent=2, ensure_ascii=False)
#     print("vision_tower config.json 已保存")