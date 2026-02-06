# Fraud Detection with LoRA

This project demonstrates how to fine-tune a lightweight Large Language Model (Qwen2.5-0.5B) for fraud detection using LoRA (Low-Rank Adaptation) on the **Kaggle Credit Card Fraud Detection** dataset.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) installed.
- Python 3.10+
- (Optional) GPU with CUDA or Mac with MPS (Apple Silicon).

## Setup

1. Initialize the project and install dependencies:
   ```bash
   # From inside the backend directory
   uv sync
   ```

## Usage

### 1. Prepare Data
Download and process the Kaggle Credit Card Fraud dataset (via Hugging Face mirror).
This script balances the dataset (since fraud is rare) and converts tabular features (V1-V28) into text descriptions for the LLM.
```bash
uv run python prepare_data.py
```
Outputs: `transactions_train.jsonl`, `transactions_val.jsonl`, `test_samples.json`.

### 2. Fine-tune the Model
Train the model using LoRA.
```bash
uv run python finetune.py
```
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: LoRA (Low-Rank Adaptation)
- Time: ~5-10 minutes (depending on hardware).

### 3. Run Inference
Test the trained model. You can pick random real samples from the test set to see if the model detects fraud correctly.
```bash
uv run python inference.py
```
*Tip: Type 'r' in the inference prompt to load a real transaction.*

## Project Structure

- `prepare_data.py`: Downloads dataset from Hugging Face, balances classes, and formats as text.
- `finetune.py`: Training script using PEFT/LoRA.
- `inference.py`: Interactive testing script.
- `pyproject.toml`: Dependency configuration.
