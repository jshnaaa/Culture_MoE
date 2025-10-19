# src/llamafactory/train/classification/callbacks.py
"""
自定义训练回调，用于保存 LoRA 权重到每个 checkpoint
"""
import os

from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    """
    在每个 checkpoint 保存时，同时保存 PEFT/LoRA 权重

    默认情况下，Hugging Face Trainer 只在训练结束时保存 PEFT 权重，
    这个回调确保每个中间 checkpoint 也包含 LoRA 权重。
    """

    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        在保存 checkpoint 时触发

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            kwargs: 包含 model 等
        """
        # 只在主进程执行
        if not state.is_world_process_zero:
            return control

        # 获取 checkpoint 目录
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        # 获取模型
        model = kwargs.get("model")
        if model is None:
            print("⚠️  Warning: Model not found in callback kwargs")
            return control

        # 检查是否是我们的 CultureMoE 模型
        if not hasattr(model, "llama_model"):
            print("⚠️  Warning: Model does not have llama_model attribute")
            return control

        llama_model = model.llama_model

        # 检查是否使用了 PEFT/LoRA
        try:
            from peft import PeftModel

            if isinstance(llama_model, PeftModel):
                # 保存 LoRA 权重
                peft_model_path = checkpoint_folder
                llama_model.save_pretrained(peft_model_path)
                print(f"✅ Saved LoRA weights to {peft_model_path}")
            else:
                print(f"ℹ️  LLaMA model is not a PeftModel, skipping LoRA save")
        except ImportError:
            print("⚠️  Warning: peft library not found, cannot save LoRA weights")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save LoRA weights: {e}")

        return control

    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        训练结束时也保存一次（确保最终模型包含 LoRA 权重）
        """
        if not state.is_world_process_zero:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        if not hasattr(model, "llama_model"):
            return control

        llama_model = model.llama_model

        try:
            from peft import PeftModel

            if isinstance(llama_model, PeftModel):
                # 保存到输出目录根目录
                final_model_path = args.output_dir
                llama_model.save_pretrained(final_model_path)
                print(f"✅ Saved final LoRA weights to {final_model_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save final LoRA weights: {e}")

        return control


class SaveFullModelCallback(TrainerCallback):
    """
    保存完整模型（包括 MoE 组件和 LoRA 权重）到每个 checkpoint
    """

    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """在保存 checkpoint 时触发"""
        if not state.is_world_process_zero:
            return control

        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        model = kwargs.get("model")
        if model is None:
            return control

        # 1. 保存 LoRA 权重
        if hasattr(model, "llama_model"):
            llama_model = model.llama_model
            try:
                from peft import PeftModel
                if isinstance(llama_model, PeftModel):
                    llama_model.save_pretrained(checkpoint_folder)
                    print(f"✅ [Checkpoint-{state.global_step}] Saved LoRA weights")
            except Exception as e:
                print(f"⚠️  [Checkpoint-{state.global_step}] Failed to save LoRA: {e}")

        # 2. 保存 tokenizer
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(checkpoint_folder)
                print(f"✅ [Checkpoint-{state.global_step}] Saved tokenizer")
            except Exception as e:
                print(f"⚠️  [Checkpoint-{state.global_step}] Failed to save tokenizer: {e}")

        return control

