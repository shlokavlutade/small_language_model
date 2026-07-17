from pathlib import Path
from .config import DEFAULT_SEED, METADATA_FILE

from .dataset_utils import (
    load_dataset_cached,
    prepare_training_dataset,
    save_dataset_to_txt,
    cleanup_temp_files
)
from .tokenizer_utils import create_bin_files
from .metadata import create_metadata

def prepare_dataset(
        experiment_name,
        output_dir,
        dataset_fraction=1.0,
        duplicate_ratio = 0.0,
        shuffle_ratio = 0.0,
        seed = DEFAULT_SEED
):
    output_dir = Path(output_dir)
    print("\nStarting experiment: ", experiment_name)

    train_dataset, val_dataset = load_dataset_cached()

    original_train_size = len(train_dataset)
    validation_size = len(val_dataset)

    processed_train = prepare_training_dataset(train_dataset, dataset_fraction,duplicate_ratio,
                                                shuffle_ratio, seed)
    final_train_size = len(processed_train)
    unique_train_size = len(set(processed_train["text"]))

    train_txt, val_txt = save_dataset_to_txt(processed_train, val_dataset)

    token_stats = create_bin_files(train_txt, val_txt, output_dir)

    create_metadata(output_file= output_dir/METADATA_FILE,
                    experiment_name=experiment_name,
                    dataset_fraction=dataset_fraction,
                    duplicate_ratio=duplicate_ratio,
                    shuffle_ratio=shuffle_ratio,
                    original_stories=original_train_size,
                    final_stories=final_train_size,
                    unique_stories=unique_train_size,
                    validation_stories=validation_size,
                    train_tokens= token_stats["train_tokens"],
                    val_tokens=token_stats["val_tokens"],
                    train_bin=token_stats["train_bin"],
                    val_bin=token_stats["val_bin"],
                    seed=seed)
    
    # cleanup_temp_files()

    print("Dataset preparation completed.")


