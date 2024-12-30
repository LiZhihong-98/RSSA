# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

# 导入训练数据集相关的方法
from psalm.train.train_datasets import *

# 导入多模态模型的配置文件解析类
from psalm.mask_config.config import Config
from fvcore.common.config import CfgNode

# 引入 Python 标准库模块 warnings，用于屏蔽警告信息
import warnings

# 忽略所有的警告信息
warnings.filterwarnings("ignore")

# 定义全局变量 local_rank，用于分布式训练时的进程标识
local_rank = None


# 定义一个函数，用于打印模型中可训练的参数模块
# 传入模型和前缀参数，打印所有 requires_grad=True 的参数
def print_trainable_parm(model, prefix):
    for name, module in model.named_modules():
        print_flag = False
        for p in module.parameters():
            if p.requires_grad == True:
                print(f"{prefix}:  {name}")
                print_flag = True
                break


# 定义函数，用于从 YAML 配置文件中加载 MaskFormer 的配置信息
def get_mask_config(
    config="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
):
    # 从配置文件路径加载
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    # 将加载的 yaml 配置合并到最终的配置信息中
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg


# 定义函数，用于检查模型中参数的数据类型（比如是否为浮点类型）
def print_dtype(model, prefix, dtype):
    for name, p in model.named_parameters():
        if p.dtype != dtype:
            print(f"{prefix}: {name}")
            print(p.dtype)


# 定义 rank0_print 方法，确保只有分布式训练的 rank0 进程打印日志
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# 定义数据模型的配置结构体
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m"
    )  # 模型的名称或路径
    version: Optional[str] = field(default="v0")  # 模型的版本
    freeze_backbone: bool = field(default=False)  # 是否冻结骨干网络
    train_backbone: bool = field(default=False)  # 是否训练骨干网络
    tune_mm_mlp_adapter: bool = field(default=False)  # 是否调整多模态 MLP 适配器
    vision_tower: Optional[str] = field(default=None)  # 视觉模型的路径或名称
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer # 选择视觉模型的哪一层作为特征输出，默认最后一层
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None
    )  # 预训练的多模态适配器路径
    mm_use_im_start_end: bool = field(default=False)  # 是否使用图像的起始和结束标记
    mm_use_im_patch_token: bool = field(default=True)  # 是否使用图像的 patch token
    mm_vision_select_feature: Optional[str] = field(
        default="patch"
    )  # 选择哪种视觉特征作为输入
    with_norm: bool = field(default=True)  # 是否在模型中使用归一化
    with_layernorm: bool = field(default=False)  # 是否在模型中使用层归一化
    skip_init_vision: bool = field(default=False)  # 是否跳过视觉模型的初始化
    with_sam: bool = field(default=False)  # 是否在模型中使用 SAM 模块
    with_swin: bool = field(default=False)  # 是否在模型中使用 Swin 模块
    with_teacher: bool = field(default=False)  # 是否在模型中使用教师模型
    swin_type: Optional[str] = field(default="base")  # Swin 模型的类型
    projector_outdim: Optional[int] = field(default=2048)  # 投影层的输出维度
    mm_projector_type: Optional[str] = field(default="swin_conv")  # 多模态投影器的类型
    model_version: Optional[str] = field(default="v1")  # 模型的版本控制
    load_mask2former: bool = field(default=True)  # 是否加载 Mask2Former 模块
    seg_task: Optional[str] = field(default="panoptic")  # 分割任务的类型
    mask_config: Optional[str] = field(
        default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )  # MaskFormer 配置文件路径
    dino_path: Optional[str] = field(default=None)  # DINO 模型的路径


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )  # 数据文件的路径
    lazy_preprocess: bool = False  # 是否延迟数据预处理
    is_multimodal: bool = False  # 数据是否为多模态
    image_folder: Optional[str] = field(default=None)  # 图像文件夹的路径
    refcoco_image_folder: Optional[str] = (
        "/path/to/refer_seg/images/mscoco/images/train2014"  # RefCOCO 数据集的图像文件夹路径
    )
    image_first: bool = field(default=True)  # 是否将图像数据优先处理
    seg_last: bool = field(default=True)  # 是否在最后处理分割任务
    instruction_version: str = "v1"  # 指令版本
    image_aspect_ratio: str = "square"  # 图像的宽高比设置
    image_grid_pinpoints: Optional[str] = field(default=None)  # 图像网格的关键点
    json_path: str = "/path/to/instruction_segmentation_train.json"  # JSON 配置文件路径
    instance_json_path: str = (
        "/path/to/instruction_segmentation_train.json"  # 实例分割的 JSON 文件路径
    )
    lvis_json_path: str = (
        "/path/to/lvis_instance_train.json"  # LVIS 数据集的 JSON 文件路径
    )
    lvis_categories_path: str = (
        "/path/to/lvis_instance_categories.json"  # LVIS 数据集的类别文件路径
    )
    region_json_path: str = (
        "/path/to/visual_prompt_segmentation_train.json"  # 区域分割的 JSON 文件路径
    )
    panoptic_json_path: str = "/path/to/coco"  # 全景分割 JSON 文件路径
    ref_coco_path: str = (
        "/path/to/refcoco/refcoco_train.json"  # RefCOCO 数据集的 JSON 路径
    )
    ref_coco_plus_path: str = (
        "/path/to/refcoco+/refcoco+_train.json"  # RefCOCO+ 数据集的 JSON 路径
    )
    ref_coco_g_path: str = (
        "/path/to/refcocog/refcocog_train.json"  # RefCOCOg 数据集的 JSON 路径
    )
    mmconv_path: str = "/path/to/llava_1_5"  # 多模态对话数据路径
    data_ratio: str = "1||1||1||1"  # 数据集之间的比例配置
    fix_dataset_len: int = 0  # 固定数据集长度
    segmentation: bool = True  # 是否启用分割任务


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)  # 缓存目录路径
    optim: str = field(default="adamw_torch")  # 优化器类型
    remove_unused_columns: bool = field(default=False)  # 是否移除未使用的列
    freeze_mm_mlp_adapter: bool = field(default=False)  # 是否冻结多模态 MLP 适配器
    mpt_attn_impl: Optional[str] = field(default="triton")  # MPT 注意力实现类型
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )  # 最大序列长度. 序列将被填充或截断.
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )  # 通过双量化统计压缩量化数据
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )  # 量化数据类型. 应为 `fp4` 或 `nf4`.
    bits: int = field(
        default=16, metadata={"help": "How many bits to use."}
    )  # 使用的位数
    lora_enable: bool = False  # 是否启用 LoRA 训练
    lora_r: int = 64  # LoRA 中 r 的参数
    lora_alpha: int = 16  # LoRA 中 alpha 的参数
    lora_dropout: float = 0.05  # LoRA 的 Dropout 比例
    lora_weight_path: str = ""  # LoRA 权重保存路径
    lora_bias: str = "none"  # LoRA 的偏置配置
    dataloader_drop_last: bool = True  # 数据加载器是否丢弃最后一批


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def make_unify_datamodule(tokenizer, data_args, training_args):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split("||")
    data_ratio = [int(data_) for data_ in data_ratio]
    panoptic_coco_dataset = COCO_panoptic_dataset_random(
        json_path=data_args.panoptic_json_path, tokenizer=tokenizer, data_args=data_args
    )
    referring_json_path = [
        data_args.ref_coco_path,
        data_args.ref_coco_plus_path,
        data_args.ref_coco_g_path,
    ]
    refcoco_dataset = RefCOCO_dataset(
        json_path=referring_json_path, tokenizer=tokenizer, data_args=data_args
    )
    region_coco_dataset = COCO_interactive_dataset(
        json_path=data_args.region_json_path, tokenizer=tokenizer, data_args=data_args
    )
    mm_conv_json = os.path.join(
        data_args.mmconv_path,
        "LLaVA-Instruct-150K/llava_v1_5_mix665k_onlyMM_filtered.json",
    )
    mm_conv_dataset = MM_Conv_Dataset(
        data_path=mm_conv_json, tokenizer=tokenizer, data_args=data_args
    )
    datasets = (
        [panoptic_coco_dataset] * data_ratio[0]
        + [refcoco_dataset] * data_ratio[1]
        + [region_coco_dataset] * data_ratio[2]
        + [mm_conv_dataset] * data_ratio[3]
    )
    print(f"the dataset ratio is: {data_ratio}")

    # you can change 16 to your frequency sets, it represents how many samples to change tasks
    train_dataset = UnifyDatasetSingleDatasetForBatch(
        datasets, data_ratio, 16, fix_dataset_len=data_args.fix_dataset_len
    )
    print(f"total unify dataset number is {len(train_dataset)}")
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    global local_rank
    # 使用 HfArgumentParser 解析命令行参数，并生成对应的 dataclass 实例
    # 解析的参数包括模型参数（ModelArguments）、数据参数（DataArguments）和训练参数（TrainingArguments）
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 将训练参数中的 local_rank 设置为全局变量，用于分布式训练时的进程标识
    local_rank = training_args.local_rank
    # 根据训练参数设置计算数据类型（如 float16、bfloat16 或 float32）
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # 从配置文件加载 MaskFormer 配置，并根据模型参数更新分割任务类型
    mask_cfg = get_mask_config(config=model_args.mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task

    # 初始化一个空的字典，用于存储从预训练中加载模型时的额外参数
    bnb_model_from_pretrained_args = {}

    # 输出日志，表示即将使用 PSALM 模型
    print("using model PSALM")
    # if not training_args.bf16:
    # 从预训练模型加载 PSALM，指定配置文件和缓存目录
    # 如果模型启用了额外参数，则将其传递给 from_pretrained 方法
    model = PSALM.from_pretrained(
        model_args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )
    # 如果未启用模型的 Mask 模块解码，则根据模型参数加载 Mask2Former 的权重
    if not model.is_train_mask_decode:
        mask2former_ckpt = (
            model_args.vision_tower if model_args.load_mask2former else None
        )
        model.initial_mask_module(mask2former_ckpt)

    # 设置模型配置参数，禁用缓存以节省内存
    model.config.use_cache = False

    # 如果冻结骨干网络，则将骨干网络的所有参数设置为不需要梯度更新
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 如果启用梯度检查点，则根据模型特性启用输入所需的梯度
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 如果模型不支持 enable_input_require_grads，则通过 hook 手动设置
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 从预训练模型加载对应的分词器（Tokenizer），并配置最大序列长度、填充方式等
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 如果分词器没有 pad_token，则添加特殊的 PAD 标记并调整嵌入层大小
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    # 根据模型版本选择默认对话模板
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    # 如果视觉模块已配置，则初始化视觉相关的模型和处理器
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )
        # 设置视觉塔的计算精度和设备
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=(
                torch.float16
                if training_args.fp16
                else (torch.bfloat16 if training_args.bf16 else torch.float32)
            ),
            device=training_args.device,
        )
        # 配置数据参数中与图像处理相关的部分
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        # 根据训练参数调整多模态适配器的梯度设置
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if not model_args.train_backbone:
            model.model.vision_tower.requires_grad_(False)

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    # 向分词器中添加特殊的 SEG 标记，并调整模型的嵌入层大小
    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    # 设置模型的特殊标记，如 SEG 和 EOS（结束标记）
    model.get_special_token(
        SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ],
        EOS=tokenizer.eos_token_id,
    )
    # 创建统一的数据模块，包括训练数据集、评估数据集和数据整理器
    data_module = make_unify_datamodule(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )
    # 确保数据加载器丢弃最后一批数据
    training_args.dataloader_drop_last = True
    # 使用 LLaVATrainer 初始化训练器，并传入模型、分词器、训练参数和数据模块
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    # 如果输出目录中存在检查点文件，则从检查点恢复训练；否则开始新训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # 保存训练状态
    trainer.save_state()

    # 训练结束后重新启用缓存功能
    model.config.use_cache = True

    # 如果启用了 LoRA，则保存 LoRA 相关的状态字典
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        # 如果未启用 LoRA，则使用安全保存方法保存模型
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
