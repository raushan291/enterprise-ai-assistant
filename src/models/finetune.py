import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config.settings import settings


class ProgressPerEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n=== Starting Epoch {state.epoch+1:.0f}/{args.num_train_epochs} ===")


os.makedirs(settings.LORA_PATH, exist_ok=True)


def load_and_prepare_data(tokenizer, max_length=512):
    dataset = load_dataset("json", data_files={"train": settings.TRAINING_DATA_PATH})

    def preprocess(example):
        model_input = tokenizer(
            example["input"],
            max_length=max_length,
            truncation=True,
        )
        labels = tokenizer(
            example["target"],
            max_length=max_length,
            truncation=True,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input

    tokenized = dataset.map(preprocess, batched=True)
    return tokenized


def run_full_evaluation():
    import gc

    from src.models.evaluator import evaluate_model, test_model

    print("\nRunning post-training evaluation for all variants...\n")

    # Load test set (subset for speed)
    eval_path = settings.EVAL_DATA_PATH
    with open(eval_path) as f:
        eval_data = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(settings.LORA_PATH)
    results_all = {}

    # Define model variants in sequence
    variants = [
        ("fp32", dict(dtype=torch.float32)),
        ("fp16", dict(dtype=torch.float16, low_cpu_mem_usage=True)),
        ("int8", dict(dtype="int8")),
    ]

    def cleanup_model(model):
        """Cleanup to release VRAM and RAM."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Iterate over each variant sequentially
    for name, opts in variants:
        print("\n\n==============================")
        print(f"Evaluating variant: {name.upper()}")
        print("==============================\n")

        model = None

        try:
            if name == "fp32":
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    settings.BASE_LLM_MODEL
                )
                model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)

            elif name == "fp16":
                try:
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        settings.BASE_LLM_MODEL,
                        torch_dtype=opts["dtype"],
                        low_cpu_mem_usage=True,
                    )
                    model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)
                except Exception as e:
                    print(f"Skipping FP16 due to error: {e}")
                    continue

            elif name == "int8":
                if torch.cuda.is_available() or torch.backends.mps.is_available():
                    print("Using BitsAndBytes quantized model for INT8...")
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        settings.BASE_LLM_MODEL,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                    model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)
                else:
                    print("Using torch.ao.quantization INT8 fallback...")
                    int8_model_file = os.path.join(
                        settings.LORA_PATH + "_int8_cpu", "adapter_model_int8.pt"
                    )
                    if os.path.exists(int8_model_file):
                        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                            settings.BASE_LLM_MODEL
                        )
                        model = PeftModel.from_pretrained(
                            base_model, settings.LORA_PATH
                        )
                        state_dict = torch.load(int8_model_file, map_location="cpu")
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        print("No INT8 fallback file found.")
                        continue

            # Run evaluation
            if model is not None:
                model.eval()
                metrics = evaluate_model(tokenizer, model, eval_data, verbose=False)

                print(f"\n=== METRICS ({name.upper()}) ===")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                print("=== DONE ===\n")

                results_all[name] = metrics

                # Run quick test example
                prompt = "What is COVID-19?"
                res = test_model(tokenizer, model, prompt)
                print(f"\n=== Test Example ({name.upper()}) ====")
                print(f"Prompt: {prompt}\nResponse: {res}")

        except Exception as e:
            print(f"Skipping {name.upper()} evaluation: {e}")

        finally:
            # Clean up model from memory before loading next model
            cleanup_model(model)

    # Save all metrics
    output_path = os.path.join(settings.LORA_PATH, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\nAll metrics saved to {output_path}")
    return results_all


def main():
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.BASE_LLM_MODEL)

    # Apply LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=settings.LORA_R,  # Rank of LoRA matrices (capacity of adaptation)
        lora_alpha=settings.LORA_ALPHA,  # Scaling factor
        target_modules=settings.LORA_TARGET_MODULES,  # Attention projection matrices
        lora_dropout=settings.LORA_DROPOUT,  # Dropout applied to LoRA layers
        bias=settings.LORA_BIAS,  # Do not train bias terms
        task_type=TaskType[
            settings.LORA_TASK_TYPE
        ],  # T5 is a sequence-to-sequence model
        fan_in_fan_out=settings.LORA_FAN_IN_FAN_OUT,  # Leave False for T5 (only True for Conv layers)
        modules_to_save=settings.LORA_MODULES_TO_SAVE,  # You can specify extra modules to keep trainable
        layers_to_transform=settings.LORA_LAYERS_TO_TRANSFORM,  # Leave None unless you want LoRA only in certain layers
        layers_pattern=settings.LORA_LAYERS_PATTERN,  # Auto-detects T5 layer patterns; safe to leave None
        inference_mode=settings.LORA_INFERENCE_MODE,  # Keep False when training (True only when loading for inference)
    )

    model = get_peft_model(model, lora_config)

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    tokenized_data = load_and_prepare_data(tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_mlflow = settings.ENABLE_MLFLOW_LOGGING

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    training_args = TrainingArguments(
        output_dir=settings.LORA_PATH,  # Directory where model checkpoints and outputs are saved
        per_device_train_batch_size=settings.PER_DEVICE_TRAIN_BATCH_SIZE,  # Number of samples processed per device during training
        per_device_eval_batch_size=settings.PER_DEVICE_EVAL_BATCH_SIZE,  # Number of samples processed per device during evaluation
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,  # Accumulate gradients to simulate larger batch size
        learning_rate=settings.LEARNING_RATE,  # Initial learning rate for optimization
        lr_scheduler_type=settings.LR_SCHEDULER_TYPE,  # Strategy for adjusting the learning rate during training
        optim=settings.OPTIMIZER,  # Optimizer used for training
        warmup_ratio=settings.WARMUP_RATIO,  # Fraction of total steps used for learning rate warmup
        num_train_epochs=settings.NUM_TRAIN_EPOCHS,  # Number of complete passes through the training dataset
        fp16=settings.USE_FP16,  # Enable 16-bit precision to reduce memory usage and speed up training
        logging_dir=f"{settings.LORA_PATH}/logs",  # Directory where training logs will be stored
        logging_steps=settings.LOGGING_STEPS,  # Log training metrics every N steps
        save_steps=settings.SAVE_STEPS,  # Save model checkpoint every N steps
        save_total_limit=settings.SAVE_TOTAL_LIMIT,  # Maximum number of checkpoints to keep
        report_to=settings.REPORT_TO,  # Disable external experiment tracking unless enabled (e.g., wandb)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.add_callback(ProgressPerEpochCallback())

    if use_mlflow:
        METRIC_KEYS = {"loss", "train_loss", "total_flos", "learning_rate", "grad_norm"}
        PARAM_KEYS = {
            "train_samples_per_second",
            "train_runtime",
            "train_steps_per_second",
        }

        mlflow.start_run(run_name=settings.MLFLOW_RUN_NAME)

        # Log hyperparameters
        mlflow.log_params(
            {
                "base_model": settings.BASE_LLM_MODEL,
                "batch_size": settings.PER_DEVICE_TRAIN_BATCH_SIZE,
                "learning_rate": settings.LEARNING_RATE,
                "epochs": settings.NUM_TRAIN_EPOCHS,
                "lora_r": settings.LORA_R,
                "lora_alpha": settings.LORA_ALPHA,
                "optimizer": settings.OPTIMIZER,
                "warmup_ratio": settings.WARMUP_RATIO,
                "gradient_accumulation_steps": settings.GRADIENT_ACCUMULATION_STEPS,
                "fp16": settings.USE_FP16,
            }
        )

        trainer.train()

        # Log loss over time
        metrics = trainer.state.log_history

        for record in metrics:
            step = int(record.get("step", record.get("epoch", 0)))
            for key, value in record.items():
                if key in METRIC_KEYS and isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value), step=step)

        # Log summary stats once
        if metrics:
            last = metrics[-1]
            summary = {k: v for k, v in last.items() if k in PARAM_KEYS}
            mlflow.log_params(summary)

        print("\nMLflow logging complete.")
    else:
        print("Training without MLflow logging...")
        trainer.train()

    print("\nSaving FP32 LoRA adapter...")
    model.save_pretrained(settings.LORA_PATH)
    tokenizer.save_pretrained(settings.LORA_PATH)

    if use_mlflow:
        mlflow.log_artifacts(settings.LORA_PATH, artifact_path="fp32_model")

    print("Converting adapter weights to FP16...")
    fp32_path = os.path.join(settings.LORA_PATH, "adapter_model.safetensors")
    fp16_path = os.path.join(settings.LORA_PATH, "adapter_model_fp16.safetensors")
    if os.path.exists(fp32_path):
        state = load_file(fp32_path)
        save_file({k: v.half() for k, v in state.items()}, fp16_path)
        print(f"FP16 adapter saved at {fp16_path}")

        if use_mlflow:
            mlflow.log_artifact(fp16_path)

    print("Preparing INT8 version for inference...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )

        base_model_8bit = AutoModelForSeq2SeqLM.from_pretrained(
            settings.BASE_LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )

        model_8bit = PeftModel.from_pretrained(base_model_8bit, settings.LORA_PATH)
        int8_path = settings.LORA_PATH + "_int8"
        model_8bit.save_pretrained(int8_path)
        tokenizer.save_pretrained(int8_path)

        print(f"INT8 adapter (bitsandbytes) saved at: {int8_path}")

        if use_mlflow:
            mlflow.log_artifacts(int8_path, artifact_path="int8_model")

    except Exception as e:
        print(f"bitsandbytes quantization failed: {e}")
        print("Falling back to CPU-safe quantization using torch.ao.quantization...")

        import torch
        from torch.ao.quantization import quantize_dynamic

        # Load model in FP32
        base_model = AutoModelForSeq2SeqLM.from_pretrained(settings.BASE_LLM_MODEL)
        model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)
        model.eval()

        # Dynamically quantize only Linear layers
        quantized_model = quantize_dynamic(model, dtype=torch.qint8)

        # Save quantized model
        int8_cpu_path = settings.LORA_PATH + "_int8_cpu"
        os.makedirs(int8_cpu_path, exist_ok=True)
        torch.save(
            quantized_model.state_dict(),
            os.path.join(int8_cpu_path, "adapter_model_int8.pt"),
        )
        tokenizer.save_pretrained(int8_cpu_path)

        print(f"INT8 (CPU) adapter saved successfully at: {int8_cpu_path}")

        if use_mlflow:
            mlflow.log_artifacts(int8_cpu_path, artifact_path="int8_cpu_model")

    metrics = run_full_evaluation()
    if use_mlflow:
        for variant, m in metrics.items():
            mlflow.log_params({f"{variant}_{k}": v for k, v in m.items()})
        mlflow.end_run()


if __name__ == "__main__":
    main()
