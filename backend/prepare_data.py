import json
import pandas as pd
import os
from datasets import load_dataset
from loguru import logger
from config import Config

class DataPreparer:
    def __init__(self):
        self.dataset_repo = Config.DATASET_REPO
        self.alt_dataset_repo = Config.ALTERNATIVE_DATASET_REPO
        
    def load_data(self):
        logger.info("Loading Credit Card Fraud dataset from Hugging Face...")
        try:
            dataset = load_dataset(self.dataset_repo, split="train")
        except Exception as e:
            logger.warning(f"Error loading primary dataset: {e}")
            logger.info("Trying alternative source...")
            dataset = load_dataset(self.alt_dataset_repo, split="train")
        return dataset.to_pandas()

    def format_transaction(self, row):
        """
        Converts a row of features (Time, Amount, V1-V28) into a text description.
        """
        features = ", ".join([f"V{i}:{row[f'V{i}']:.2f}" for i in range(1, 29)])
        return f"Time: {row['Time']}, Amount: {row['Amount']:.2f}, Features: {features}"

    def balance_data(self, df):
        df_fraud = df[df['Class'] == 1]
        df_normal = df[df['Class'] == 0]
        
        logger.info(f"Fraud cases: {len(df_fraud)}")
        logger.info(f"Normal cases: {len(df_normal)}")
        
        # Sample normal cases to be 2x the number of fraud cases
        df_normal_sample = df_normal.sample(n=len(df_fraud) * 2, random_state=42)
        
        df_balanced = pd.concat([df_fraud, df_normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df_balanced, df_fraud, df_normal

    def save_to_jsonl(self, dataframe, filename):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w") as f:
            for _, row in dataframe.iterrows():
                description = self.format_transaction(row)
                label = "Fraud" if row['Class'] == 1 else "Normal"
                
                entry = {
                    "messages": [
                        {"role": "user", "content": f"Analyze this transaction for fraud: {description}"},
                        {"role": "assistant", "content": label}
                    ]
                }
                f.write(json.dumps(entry) + "\n")

    def run(self):
        df = self.load_data()
        logger.info(f"Total rows: {len(df)}")
        
        df_balanced, df_fraud, df_normal = self.balance_data(df)
        logger.info(f"Balanced dataset size: {len(df_balanced)}")
        
        # Split into Train and Validation
        val_size = int(len(df_balanced) * 0.1)
        df_train = df_balanced[:-val_size]
        df_val = df_balanced[-val_size:]
        
        logger.info(f"Saving {Config.TRAIN_FILE}...")
        self.save_to_jsonl(df_train, Config.TRAIN_FILE)
        
        logger.info(f"Saving {Config.VAL_FILE}...")
        self.save_to_jsonl(df_val, Config.VAL_FILE)
        
        # Save test samples
        test_samples = pd.concat([
            df_fraud.sample(5, random_state=1), 
            df_normal.sample(5, random_state=1)
        ]).sample(frac=1)
        
        # Ensure directory exists for test samples too
        os.makedirs(os.path.dirname(Config.TEST_SAMPLES_FILE), exist_ok=True)
        test_samples.to_json(Config.TEST_SAMPLES_FILE, orient="records")
        logger.info(f"Saved {Config.TEST_SAMPLES_FILE} for inference script.")
        logger.success("Data preparation completed successfully!")

if __name__ == "__main__":
    preparer = DataPreparer()
    preparer.run()
