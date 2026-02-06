import os

class Config:
    # Model Configuration
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    NEW_MODEL_NAME = "Qwen2.5-0.5B-Fraud-Detection"
    
    # Dataset Configuration
    DATASET_REPO = "David-Egea/Creditcard-fraud-detection"
    ALTERNATIVE_DATASET_REPO = "codesignal/credit-card-fraud-detection"
    TRAIN_FILE = "dataset/transactions_train.jsonl"
    VAL_FILE = "dataset/transactions_val.jsonl"
    TEST_SAMPLES_FILE = "dataset/test_samples.json"
    
    # LoRA Configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LORA_BIAS = "none"
    
    # Training Configuration
    OUTPUT_DIR = "./results"
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 10
    EVAL_STEPS = 50
    SAVE_STEPS = 100
    
    # Inference Configuration
    MAX_NEW_TOKENS = 10
    
    @classmethod
    def get_device(cls):
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def get_torch_dtype(cls):
        import torch
        device = cls.get_device()
        return torch.float16 if device != "cpu" else torch.float32
