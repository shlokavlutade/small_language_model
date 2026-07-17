from pathlib import Path
import numpy as np
import tiktoken

from .config import TOKENIZER_NAME


def get_tokenizer():
    """Load GPT-2 tokenizer using tiktoken"""
    
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    return tokenizer


def tokenize_file(input_file, output_file):

    tokenizer = get_tokenizer()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    token_buffer = []

    print(f"Reading and tokenizing: {input_file}")

    with open(output_file, "wb") as fout:

        with open(input_file, "r", encoding="utf-8") as fin:

            for idx, line in enumerate(fin):

                tokens = tokenizer.encode(
                    line,
                    allowed_special={"<|endoftext|>"}
                )

                token_buffer.extend(tokens)

                if len(token_buffer) >= 1_000_000:

                    np.array(
                        token_buffer,
                        dtype=np.uint16
                    ).tofile(fout)

                    total_tokens += len(token_buffer)
                    token_buffer = []

                if idx % 100000 == 0:
                    print(f"Processed {idx:,} stories")

            if token_buffer:

                np.array(
                    token_buffer,
                    dtype=np.uint16
                ).tofile(fout)

                total_tokens += len(token_buffer)

    print(f"Saved: {output_file}")
    print(f"Total tokens: {total_tokens:,}")

    return total_tokens



def create_bin_files(
        train_txt,
        val_txt,
        output_dir
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"


    print("\nTokenizing training data")

    train_tokens = tokenize_file(
        train_txt,
        train_bin
    )


    print("\nTokenizing validation data")

    val_tokens = tokenize_file(
        val_txt,
        val_bin
    )


    print("\nBinary files created successfully")


    return {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin)
    }