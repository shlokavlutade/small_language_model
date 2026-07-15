from pathlib import Path
import numpy as np
import tiktoken

from .config import TOKENIZER_NAME

def get_tokenizer():
    """Load GPT- 2 tokenizer using tiktoken"""

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    return tokenizer

def load_text_file(filepath):

    with open(filepath, "r",encoding="utf-8") as f:
        return f.read()
    

def tokenize_text(text, tokenizer):
    tokens = tokenizer.encode(text, allowed_special = {"<|endoftext|>"}
                              )
    return tokens

def save_tokens_to_bin(tokens, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.array(tokens, dtype=np.uint16).tofile(output_file)
    print(f"Saved: {output_file}")

def create_bin_files(train_txt,val_txt,output_dir):

    tokenizer = get_tokenizer()

    print("Tokenizing training data")
    
    train_text = load_text_file(train_txt)
    train_tokens = tokenize_text(train_text, tokenizer)

    print("Tokenize validation data")

    val_text = load_text_file(val_txt)
    val_tokens = tokenize_text(val_text,tokenizer)
    
    print(f"Training tokens length : {len(train_tokens)}")
    print(f"Validation token length : {len(val_tokens)}")

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok= True)

    train_bin = (output_dir/ "train.bin")
    val_bin = (output_dir/ "val.bin")

    save_tokens_to_bin(train_tokens, train_bin)
    save_tokens_to_bin(val_tokens, val_bin)

    return{
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_bin": str(train_bin),
        "val_bin": str(val_bin)
    }

