import json
from datetime import datetime
from pathlib import Path


def create_metadata(output_file,
                    experiment_name,
                    dataset_fraction,
                    duplicate_ratio,
                    shuffle_ratio,
                    original_stories,
                    final_stories,
                    unique_stories,
                    validation_stories,
                    train_tokens,
                    val_tokens,
                    train_bin,
                    val_bin,
                    seed):
    metadata = {
        "experiment_name": experiment_name,
        "parameters":{
            "dataset_fraction": dataset_fraction,
            "duplicate_ratio": duplicate_ratio,
            "shuffle_ratio": shuffle_ratio,
            "seed": seed
        },
        "dataset_stats": {
            "original_train_stories": original_stories,
            "final_train_stories": final_stories,
            "unique_train_stories": unique_stories,
            "validation_stories": validation_stories
        },
        "token_stats":{
            "train_tokens": train_tokens,
            "validation_tokens": val_tokens
        },
        "files":{
            "train_bin": str(train_bin),
            "val_bin": str(val_bin)
        }
    }
    
    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )
    print(f"Saved metadata : {output_file}")

