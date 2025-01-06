# 通过对比基准模型和目标模型的参数，生成一个仅包含两者差异的差异模型（delta model），以显著减少存储和传输的成本。生成的差异模型可以保存到本地，或者上传到 Hugging Face Hub 等平台，方便其他用户基于基准模型快速复现目标模型，从而提升模型共享和协作的效率。

"""
Usage:
python3 -m psalm.model.make_delta --base ~/model_weights/llama-7b --target ~/model_weights/llava-7b --delta ~/model_weights/llava-7b-delta --hub-repo-id liuhaotian/llava-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from psalm.model.utils import auto_upgrade


def make_delta(base_model_path, target_model_path, delta_path, hub_repo_id):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Loading target model")
    auto_upgrade(target_model_path)
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Calculating delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        if name not in base.state_dict():
            assert name in [
                "model.mm_projector.weight",
                "model.mm_projector.bias",
            ], f"{name} not in base model"
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        else:
            assert name in [
                "model.embed_tokens.weight",
                "lm_head.weight",
            ], f"{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}"
            bparam = base.state_dict()[name]
            param.data[: bparam.shape[0], : bparam.shape[1]] -= bparam

    print("Saving delta")
    if hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": hub_repo_id}
    else:
        kwargs = {}
    target.save_pretrained(delta_path, **kwargs)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    args = parser.parse_args()

    make_delta(
        args.base_model_path, args.target_model_path, args.delta_path, args.hub_repo_id
    )
