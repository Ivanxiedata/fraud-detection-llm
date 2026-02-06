import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from loguru import logger
from config import Config

class FraudDetectionTrainer:
    def __init__(self):
        self.device = Config.get_device()
        self.model_name = Config.MODEL_NAME
        self.new_model_name = Config.NEW_MODEL_NAME
        logger.info(f"Using device: {self.device}")

    def load_model_and_tokenizer(self):
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=Config.get_torch_dtype(),
            device_map=self.device,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def get_lora_config(self):
        return LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias=Config.LORA_BIAS,
            task_type=TaskType.CAUSAL_LM,
            target_modules=Config.LORA_TARGET_MODULES
        )

    def load_datasets(self):
        logger.info("Loading dataset...")
        return load_dataset("json", data_files={"train": Config.TRAIN_FILE, "validation": Config.VAL_FILE})

    def get_training_args(self):
        return TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            logging_steps=Config.LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=Config.EVAL_STEPS,
            save_strategy="steps",
            save_steps=Config.SAVE_STEPS,
            fp16=(self.device == "cuda"),
            bf16=(self.device == "mps"),
            # use_mps_device=(self.device == "mps"),
            report_to="none"
        )

    def train(self):
        model, tokenizer = self.load_model_and_tokenizer()
        peft_config = self.get_lora_config()
        dataset = self.load_datasets()
        training_args = self.get_training_args()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Saving model to {self.new_model_name}...")
        trainer.model.save_pretrained(self.new_model_name)
        tokenizer.save_pretrained(self.new_model_name)
        logger.success("Training completed and model saved!")

if __name__ == "__main__":
    trainer = FraudDetectionTrainer()
    trainer.train()
