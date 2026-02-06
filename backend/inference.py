import torch
import json
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger
from config import Config

class FraudDetectionInference:
    def __init__(self):
        self.device = Config.get_device()
        self.base_model_name = Config.MODEL_NAME
        self.adapter_path = Config.NEW_MODEL_NAME
        self.model = None
        self.tokenizer = None
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=Config.get_torch_dtype(),
            device_map=self.device,
            trust_remote_code=True
        )

        logger.info("Loading LoRA adapter...")
        if not os.path.exists(self.adapter_path):
            logger.warning(f"Adapter path '{self.adapter_path}' not found. Using base model only.")
            self.model = base_model
        else:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def load_test_samples(self):
        if os.path.exists(Config.TEST_SAMPLES_FILE):
            with open(Config.TEST_SAMPLES_FILE, "r") as f:
                return json.load(f)
        return []

    def format_transaction_string(self, row_dict):
        features = ", ".join([f"V{i}:{row_dict.get(f'V{i}', 0):.2f}" for i in range(1, 29)])
        return f"Time: {row_dict.get('Time', 0)}, Amount: {row_dict.get('Amount', 0):.2f}, Features: {features}"

    def predict(self, description):
        messages = [
            {"role": "user", "content": f"Analyze this transaction for fraud: {description}"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def run_interactive(self):
        self.load_model()
        test_samples = self.load_test_samples()

        print("\nFraud Detection Model Ready!")
        print("Options:")
        print("1. Type 'r' to pick a random real transaction from the test set.")
        print("2. Type a custom transaction description.")
        print("3. Type 'quit' to exit.")
        print("-" * 50)

        while True:
            # Interactive prompts should still use input/print for UI
            user_input = input("\nInput: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            description = user_input
            true_label = None

            if user_input.lower() == "r":
                if test_samples:
                    sample = random.choice(test_samples)
                    description = self.format_transaction_string(sample)
                    true_label = "Fraud" if sample.get("Class") == 1 else "Normal"
                    print(f"\n[Selected Sample Data]\nLabel: {true_label}")
                    print(f"Content: {description[:100]}...")
                else:
                    logger.warning(f"No test samples found in {Config.TEST_SAMPLES_FILE} (run prepare_data.py first).")
                    continue

            response = self.predict(description)
            print(f"Model Prediction: {response}")
            
            if true_label:
                result = "CORRECT" if true_label.lower() in response.lower() else "WRONG"
                print(f"Verification: {result}")

if __name__ == "__main__":
    inference = FraudDetectionInference()
    inference.run_interactive()
