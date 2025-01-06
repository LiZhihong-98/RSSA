import os
import torch
import shutil
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.distributed as dist
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_torch_tpu_available,
    is_accelerate_available,
)

# 如果 APEX 库可用，导入它
if is_apex_available():
    from apex import amp

# 如果 SageMaker MP 环境可用，导入相关工具
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

# 导入一系列需要的库和工具，包括上下文管理器、路径处理、随机操作、以及 transformers 的组件
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import (
    PreTrainedModel,
    load_sharded_checkpoint,
    unwrap_model,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

# 默认回调和进度回调
DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

# 定义检查点文件名的常量
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

# 工具函数：可能需要对 Zero-3 优化参数进行特殊处理
# "Zero-3" 是 DeepSpeed 中的一种分布式内存优化技术


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# 提取并保存多模态适配器状态
# 主要用于处理特定关键字的参数


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


# 自定义 Trainer 类，继承自 transformers 的 Trainer
class LLaVATrainer(Trainer):

    # 重写保存检查点的方法，根据条件保存模型的多模态适配器状态
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # 仅保存多模态适配器的权重
            keys_to_match = ["mm_projector"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    # 重写保存方法，支持保存的自定义操作
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # 更新历史损失字典，用于记录不同损失类型的历史值
    def update_history_loss_dict(self, outputs):
        if not hasattr(self, "history_loss_dict"):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if "loss" in name and name != "loss":
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()

    # 自定义计算损失的方法
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        定义 Trainer 的损失计算逻辑，支持自定义行为。
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # 如果模型返回 "past" 状态，保存下来（通常用于语言模型）
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if (
                unwrap_model(model)._get_name()
                in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            ):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # 如果输出是字典，从中提取损失值
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and "loss_dice" in outputs:
                loss_dict = {}
                for name, value in outputs.items():
                    if "loss" in name and name != "loss":
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self, "history_loss_dict"):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss
