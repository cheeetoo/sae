import datasets
from einops import rearrange
import transformer_lens
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = transformer_lens.HookedTransformer.from_pretrained(
    "google/gemma-2-2b", device=device
)
model.eval()
model.compile()


def map_fn(batch, model=model):
    toks = model.to_tokens(batch["text"])

    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
        acts = cache["resid_post", 13].squeeze(0).cpu()

    return acts


print("Loading dataset...")
dataset = datasets.load_dataset("NeelNanda/pile-10k")

print("Getting activations...")
dataset = dataset.map(
    map_fn,
    batched=True,
    batch_size=256,
    fn_kwargs={"model": model},
)

print("Preparing dataset...")
acts = torch.cat(dataset, dim=0)  # type: ignore
acts = rearrange(acts, "b t d -> (b t) d")

n = acts.shape[1]
mean_squared_norm = torch.mean(torch.sum(acts**2, dim=1))
scaling_factor = (n / mean_squared_norm) ** 0.5
activations_tensor = acts * scaling_factor

indices = torch.randperm(len(activations_tensor))
activations_tensor = activations_tensor[indices]

print("Saving dataset...")
torch.save(activations_tensor, "pile_10k_activations.pt")
print(f"Dataset saved with {len(activations_tensor)} activation vectors.")
