import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
NUM_PROC = 4
TOKENIZER_NAME = "gpt2"
VAL_RATIO = 0.0005
SEED = 2357
DTYPE = np.uint16
BATCH_SIZE = 1024  

# -----------------------------
# Tokenizer
# -----------------------------
enc = tiktoken.get_encoding(TOKENIZER_NAME)
EOT = enc.eot_token


def tokenize_batch(examples):
    ids_batch = []
    lens = []

    for text in examples["text"]:
        ids = enc.encode_ordinary(text) # do not add special tokens
        ids.append(EOT)
        ids_batch.append(ids)
        lens.append(len(ids))

    return {"ids": ids_batch, "len": lens}


if __name__ == "__main__":
    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = load_dataset(
        "/root/vermind/data/openwebtext",
        num_proc=NUM_PROC,
    )

    split_dataset = dataset["train"].train_test_split(
        test_size=VAL_RATIO,
        seed=SEED,
        shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    # -----------------------------
    # Tokenize (batched, faster)
    # -----------------------------
    tokenized = split_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["text"],
        num_proc=NUM_PROC,
        desc="tokenizing",
    )

    # -----------------------------
    # Write binary files
    # -----------------------------
    out_dir = os.path.dirname(__file__)

    for split, dset in tokenized.items():
        print(f"\nProcessing split: {split}")

        total_tokens = int(np.sum(dset["len"], dtype=np.uint64))
        bin_path = os.path.join(out_dir, f"{split}.bin")

        arr = np.memmap(
            bin_path,
            dtype=DTYPE,
            mode="w+",
            shape=(total_tokens,),
        )

        idx = 0
        for ex in tqdm(dset, desc=f"writing {split}.bin"):
            ids = ex["ids"]
            arr[idx: idx + len(ids)] = ids
            idx += len(ids)

        assert idx == total_tokens, f"token count mismatch in {split}"
        arr.flush()

        # optional but strongly recommended
        with open(os.path.join(out_dir, f"{split}.meta"), "w") as f:
            f.write(f"tokens={total_tokens}\n")
            f.write(f"dtype={DTYPE.__name__}\n")
            f.write(f"eot={EOT}\n")
            f.write(f"tokenizer={TOKENIZER_NAME}\n")

        print(f"{split}: {total_tokens:,} tokens written")
