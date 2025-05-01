from typing import List, Dict

import datasets
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoConfig  # type: ignore

MODEL_NAME = "google/gemma-2-2b"
LOCAL_DIR = "tokenized_fineweb"
REPO_ID = "cheeetoo/fineweb-tokenized-gemma-2"
DATASET_NAME = "HuggingfaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"


print(f"Loading tokenizer for '{MODEL_NAME}' …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(MODEL_NAME)
seq_len: int | None = getattr(config, "max_position_embeddings", None)

print(f"Loading dataset: {DATASET_NAME} ({DATASET_CONFIG}) …")
dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG)


def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    return {"input_ids": encodings["input_ids"]}


tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=64,
    remove_columns=["text"],
    desc="Tokenizing",
)

tokenized_dataset.set_format("torch", columns=["input_ids"])


LOCAL_DIR.mkdir(exist_ok=True, parents=True)
print(f"Saving tokenized dataset to '{LOCAL_DIR.resolve()}' …")
tokenized_dataset.save_to_disk(str(LOCAL_DIR))

print(f"Uploading to the Hugging Face Hub at '{REPO_ID}' …")

api = HfApi()

api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=True)

api.upload_folder(
    folder_path=str(LOCAL_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add tokenized FineWeb dataset",
)

print("✅ Upload complete!")
