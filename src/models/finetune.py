import os

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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

    trainer.train()

    model.save_pretrained(settings.LORA_PATH)
    tokenizer.save_pretrained(settings.LORA_PATH)


if __name__ == "__main__":
    main()
