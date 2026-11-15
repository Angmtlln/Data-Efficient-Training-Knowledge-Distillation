"""Pre-download models into the HF cache so experiments run offline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download

from src.config import EMBED_MODEL_NAME, MODEL_NAME, PPL_MODEL_NAME

if __name__ == "__main__":
    for name in (MODEL_NAME, EMBED_MODEL_NAME, PPL_MODEL_NAME):
        path = snapshot_download(
            name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "1_Pooling/*"]
        )
        print(f"{name} -> {path}")
