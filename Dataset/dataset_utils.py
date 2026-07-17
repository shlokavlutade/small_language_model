import random
import re
from pathlib import Path

from datasets import load_dataset, Dataset

from .config import(
    HF_DATASET_NAME,
    HF_CACHE_DIR,
    TRAIN_TXT,
    VAL_TXT,
    TEMP_DIR,
    DEFAULT_SEED
)

def load_dataset_cached():
    """
    Load the dataset from Hugging Face cache or download it if not present.
    """
    print("Loading TinyStories dataset from Hugging Face...")
    train_dataset = load_dataset(HF_DATASET_NAME, split="train", cache_dir=HF_CACHE_DIR)
    val_dataset = load_dataset(HF_DATASET_NAME, split="validation", cache_dir=HF_CACHE_DIR)
    
    print(f"Training stories length: {len(train_dataset)}")
    print(f"Validation stories length: {len(val_dataset)}")

    return train_dataset, val_dataset

def reduce_dataset(dataset, fraction: float, seed=DEFAULT_SEED):
    """
    Reduce the size of the dataset to a specified fraction.
    """

    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("Fraction must be > 0.0")
    
    total_size = len(dataset)
    selected_size = int(total_size * fraction)

    random.seed(seed)

    indices =random.sample(range(total_size), selected_size)
    indices.sort()
    reduced_dataset = dataset.select(indices)

    print(f"Dataset reduced: " 
          f"{total_size:,} -> {len(reduced_dataset):,}")
    
    return reduced_dataset

def duplicate_dataset( dataset, duplicate_ratio, seed=DEFAULT_SEED):
    """ 
    Apply duplication while keeping the final dataset size same
    """
    if duplicate_ratio == 0:
        return dataset
    
    random.seed(seed)

    dataset_size = len(dataset)

    if duplicate_ratio == 0.5:
        unique_count = int(dataset_size * 0.75)

    elif duplicate_ratio == 1.0:
        unique_count = int(dataset_size * 0.50)

    else:
        raise ValueError(" Allowed duplication ratios: 0, 0.5, 1.0")
    
    unique_indices = random.sample(
        range(dataset_size), unique_count)

    duplicate_count = dataset_size - unique_count

    unique_dataset = dataset.select(unique_indices)

    duplicate_indices = random.choices(
        unique_indices,
        k= duplicate_count
    )

    duplicate_samples = dataset.select(
        duplicate_indices
    )

    combined_dataset = Dataset.from_dict(
        {"text":
        unique_dataset["text"] + duplicate_samples["text"]}
    ).shuffle(seed=seed)

    print(f"Duplication Applied: ", f"{len(unique_dataset)} unique and ",
          f"{len(duplicate_samples)} duplicates")
    
    return combined_dataset

def split_sentences(text):
    """
    Simple sentence splitter.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]



def shuffle_story_sentences(
        dataset,
        shuffle_ratio,
        seed=DEFAULT_SEED
):
    """
    Randomly shuffle sentences in a percentage
    of stories.
    """

    if shuffle_ratio == 0:
        return dataset


    random.seed(seed)

    dataset_size = len(dataset)

    number_to_shuffle = int(
        dataset_size * shuffle_ratio
    )


    indices = random.sample(
        range(dataset_size),
        number_to_shuffle
    )


    modified_texts = dataset["text"].copy()


    for idx in indices:

        sentences = split_sentences(
            modified_texts[idx]
        )

        if len(sentences) > 1:

            random.shuffle(sentences)

            modified_texts[idx] = " ".join(
                sentences
            )


    new_dataset = Dataset.from_dict(
        {
            "text": modified_texts
        }
    )


    print(
        f"Sentence shuffling applied to "
        f"{number_to_shuffle:,} stories"
    )


    return new_dataset


def prepare_training_dataset(
        train_dataset,
        dataset_fraction=1.0,
        duplicate_ratio=0.0,
        shuffle_ratio=0.0,
        seed=DEFAULT_SEED
):
    """
    Complete training dataset transformation pipeline.

    Order:
        1. Reduce dataset
        2. Duplicate dataset
        3. Shuffle sentences
    """

    dataset = reduce_dataset(
        train_dataset,
        dataset_fraction,
        seed
    )


    dataset = duplicate_dataset(
        dataset,
        duplicate_ratio,
        seed
    )


    dataset = shuffle_story_sentences(
        dataset,
        shuffle_ratio,
        seed
    )


    return dataset


def save_dataset_to_txt(
        train_dataset,
        val_dataset
):

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        TRAIN_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        for story in train_dataset["text"]:

            f.write(
                " ".join(story.strip().split())
                +
                "\n<|endoftext|>\n"
            )


    with open(
        VAL_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        for story in val_dataset["text"]:

            f.write(
                " ".join(story.strip().split())
                +
                "\n<|endoftext|>\n"
            )

    print("Temporary text files created.")

    return TRAIN_TXT, VAL_TXT


def cleanup_temp_files():

    for file in [TRAIN_TXT, VAL_TXT]:

        if Path(file).exists():

            Path(file).unlink()


    print("Temporary files deleted.")