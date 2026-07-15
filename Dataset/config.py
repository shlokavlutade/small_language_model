from pathlib import Path

DEFAULT_SEED = 42

HF_DATASET_NAME = "roneneldan/TinyStories"
HF_CACHE_DIR = "./hf_cache"

TOKENIZER_NAME = "gpt2"

TEMP_DIR = Path("/tmp/tinystories")
TRAIN_TXT = TEMP_DIR / "train.txt"
VAL_TXT = TEMP_DIR / "validation.txt"

TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
METADATA_FILE = "metadata.json"

DATASET_FRACTIONS = [0.25, 0.50, 1.00]
DUPLICATION_RATIOS = [0.0, 0.5, 1.0]
SHUFFLE_RATIOS = [0.0, 0.25, 0.5]