#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
import torch
from psalm.model import *

from psalm.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from psalm.train.train_datasets import get_mask_config
from psalm.model.language_model.llava_phi import PSALM, PSALMForDAVISEval


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_args,
    mask_config="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
):
    """
    加载预训练模型的函数。

    参数：
    - model_path: 模型的路径。
    - model_base: 基础模型路径。
    - model_name: 模型名称。
    - model_args: 模型参数。
    - mask_config: mask配置文件路径，默认为指定路径。
    - load_8bit: 是否加载8位量化模型。
    - load_4bit: 是否加载4位量化模型。
    - device_map: 指定设备映射（默认"auto"）。
    - device: 设备类型，默认使用"cuda"。

    返回值：
    - tokenizer: 加载的分词器。
    - model: 加载的模型实例。
    - image_processor: 图像处理器实例。
    - context_len: 模型上下文长度。
    """

    # 加载模型的设备映射配置
    kwargs = {"device_map": "cpu"}

    # 如果选择加载8位量化模型，添加相应参数
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        # 如果选择加载4位量化模型，添加量化配置参数
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        # 默认使用 float16 数据类型
        kwargs["torch_dtype"] = torch.float16

    print("loading segmentation model")

    # 定义模型名称和类之间的映射关系
    model_map = {"psalm": PSALM, "psalm_video": PSALMForDAVISEval}

    # 根据提供的模型参数获取模型名称
    model_map_name = model_args.model_map_name

    # 加载 mask 配置文件
    mask_cfg = get_mask_config(mask_config)

    # 根据模型参数设置分割任务类型
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = (
        model_args.seg_task if hasattr(model_args, "seg_task") else "instance"
    )

    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    print(f"current model is {model_map_name}")

    # 根据模型名称加载对应的模型类
    model = model_map[model_map_name].from_pretrained(
        model_path, mask_decoder_cfg=mask_cfg, **kwargs
    )

    # 获取模型的视觉分支（vision tower）
    vision_tower = model.get_vision_tower()

    # 如果视觉分支未加载，可以选择性加载模型
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()

    # 将视觉分支加载到指定设备
    # vision_tower.to(device=device, dtype=torch.float16)
    vision_tower.to(device=device)

    # 如果图像处理器是字典类型，提取其中的实例分支处理器
    # if isinstance(vision_tower.image_processor, dict):
    #     image_processor = vision_tower.image_processor['instance']
    # else:
    image_processor = vision_tower.image_processor

    # 检查模型配置是否包含上下文长度信息
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        # 默认上下文长度为2048
        context_len = 2048

    return tokenizer, model, image_processor, context_len
