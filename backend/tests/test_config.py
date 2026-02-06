import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def test_config_values():
    assert Config.MODEL_NAME == "Qwen/Qwen2.5-0.5B-Instruct"
    assert Config.LORA_R == 16
    assert Config.LORA_ALPHA == 32
    assert Config.LORA_DROPOUT == 0.05
    assert Config.LORA_BIAS == "none"
    assert Config.NUM_EPOCHS == 1

def test_device_detection():
    device = Config.get_device()
    assert device in ["cuda", "mps", "cpu"]

def test_torch_dtype():
    import torch
    dtype = Config.get_torch_dtype()
    assert dtype in [torch.float16, torch.float32]
