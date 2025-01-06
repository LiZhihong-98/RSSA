# 检查指定的模型配置文件是否需要从旧版本（如 LLaMA）升级到新版本（如 LLaVA）。如果检测到需要升级，它会提示用户确认，并根据新版本的要求更新模型的类型和架构名称。升级后的配置文件会被保存，用于兼容新的代码基准。如果用户拒绝升级，程序将终止运行。这一过程确保了旧版本检查点能够正确适配新的代码和框架。
from transformers import AutoConfig


def auto_upgrade(config):
    # 从预训练配置文件加载模型的配置信息
    cfg = AutoConfig.from_pretrained(config)

    # 检查配置路径中是否包含"llava"，并确认模型类型是否需要升级
    if "llava" in config and "llava" not in cfg.model_type:
        # 确保当前模型类型是"llama"，说明这是旧版本的 LLaVA 检查点
        assert cfg.model_type == "llama"

        # 提示用户需要将旧版本检查点升级到新的代码基准
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )

        # 要求用户确认是否执行升级
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")

        if confirm.lower() in ["y", "yes"]:
            # 用户确认升级，开始升级过程
            print("Upgrading checkpoint...")

            # 确保模型架构只有一个（旧版本检查点的一致性检查）
            assert len(cfg.architectures) == 1

            # 动态修改配置的模型类型为"llava"
            setattr(cfg.__class__, "model_type", "llava")

            # 更新模型的架构名称为新的架构"LlavaLlamaForCausalLM"
            cfg.architectures[0] = "LlavaLlamaForCausalLM"

            # 保存升级后的配置文件
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            # 用户拒绝升级，终止程序
            print("Checkpoint upgrade aborted.")
            exit(1)
