from pathlib import Path

DEFAULT_SEED = 42

HF_DATASET_NAME = "roneneldan/TinyStories"
HF_CACHE_DIR = "/kaggle/working/hf_cache"

TOKENIZER_NAME = "gpt2"

TEMP_DIR = Path("/tmp/tinystories")
TRAIN_TXT = TEMP_DIR / "train.txt"
VAL_TXT = TEMP_DIR / "validation.txt"

TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
METADATA_FILE = "metadata.json"

DATASET_OUTPUT_DIR = Path("/kaggle/working/tinystories-thesis-dataset")

DATASET_FRACTIONS = [0.25, 0.50, 1.00]
DUPLICATION_RATIOS = [0.0, 0.5, 1.0]
SHUFFLE_RATIOS = [0.0, 0.25, 0.5]

EXPERIMENT_CONFIGS = {

    "run1_baseline": {
        "dataset_fraction": 1.0,
        "duplicate_ratio": 0.0,
        "shuffle_ratio": 0.0
    },

    "run2_size25": {
        "dataset_fraction": 0.25,
        "duplicate_ratio": 0.0,
        "shuffle_ratio": 0.0
    },

    "run3_size50": {
        "dataset_fraction": 0.50,
        "duplicate_ratio": 0.0,
        "shuffle_ratio": 0.0
    },

    "run4_dup50": {
        "dataset_fraction": 1.0,
        "duplicate_ratio": 0.5,
        "shuffle_ratio": 0.0
    },

    "run5_dup100": {
        "dataset_fraction": 1.0,
        "duplicate_ratio": 1.0,
        "shuffle_ratio": 0.0
    },

    "run6_noise25": {
        "dataset_fraction": 1.0,
        "duplicate_ratio": 0.0,
        "shuffle_ratio": 0.25
    },

    "run7_noise50": {
        "dataset_fraction": 1.0,
        "duplicate_ratio": 0.0,
        "shuffle_ratio": 0.5
    }
}