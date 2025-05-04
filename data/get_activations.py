import torch
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset, Dataset
from einops import rearrange
from tqdm.auto import tqdm

MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
OUTPUT_FILE = "pile_activations.pt"
REPO_ID = "cheeetoo/gemma-2-2b-fineweb-l13-acts"
BS = 40
LAYER = 13

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {MODEL_NAME}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
model.compile()
model.cfg.n_ctx = model.cfg.n_ctx // 32

print(f"Loading dataset {DATASET_NAME} ({DATASET_CONFIG})")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train[:1%]")


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BS,
)

acts = []

for batch in tqdm(dataloader):

    def hook(x, hook):
        act = rearrange(x.detach().to("cpu"), "b t d -> (b t) d")
        acts.append(act)

    with torch.no_grad():
        toks = model.to_tokens(batch["text"], truncate=True, prepend_bos=False)
        model.run_with_hooks(
            toks,
            fwd_hooks=[(utils.get_act_name("resid_post", LAYER), hook)],
            clear_contexts=True
        )
    torch.cuda.empty_cache()

print("Preparing dataset...")
acts = torch.cat(acts, dim=0)

n = acts.shape[1]
mean_squared_norm = torch.mean(torch.sum(acts**2, dim=1))
scaling_factor = (n / mean_squared_norm) ** 0.5
activations_tensor = acts * scaling_factor

indices = torch.randperm(len(activations_tensor))
activations_tensor = activations_tensor[indices]

print("Saving dataset locally...")
torch.save(activations_tensor, OUTPUT_FILE)
print(f"Dataset saved with {len(activations_tensor)} activation vectors.")

print(f"Uploading activations to Hugging Face Hub at '{REPO_ID}'...")
hf_dataset = Dataset.from_dict(
    {
        "activations": activations_tensor.detach().cpu().numpy(),
        "d_model": [n],
        "model_name": [MODEL_NAME],
        "layer": [LAYER],
        "dataset_name": [DATASET_NAME],
    }
)

hf_dataset.push_to_hub(
    repo_id=REPO_ID,
    commit_message=f"Upload activations from {MODEL_NAME} layer {LAYER}",
)

print("✅ Upload to Hugging Face Hub complete!")
