import datasets
from einops import rearrange
import transformer_lens
import transformer_lens.utils as utils
import torch

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)

LAYER = 13

model = transformer_lens.HookedTransformer.from_pretrained(
    "google/gemma-2-2b", device=device
)
model.eval()
model.compile()

print("Loading dataset...")
dataset = datasets.load_dataset("HuggingfaceFW/fineweb", "sample-10BT")

print("Getting activations...")
acts = []


def map_fn(batch, model):
    def hook(x, hook):
        act = rearrange(x.clone().cpu(), "b t d -> (b t) d")
        acts.append(act)

    toks = model.to_tokens(batch["text"], truncate=True).to(device)

    with torch.no_grad():
        model.run_with_hooks(
            toks,
            fwd_hooks=[
                (utils.get_act_name("resid_post", LAYER), hook),
            ],
        )


dataset = dataset.map(
    map_fn, batched=True, batch_size=1, fn_kwargs={"model": model}, num_proc=1
)

print("Preparing dataset...")
acts = torch.cat(acts, dim=0)  # type: ignore

n = acts.shape[1]
mean_squared_norm = torch.mean(torch.sum(acts**2, dim=1))
scaling_factor = (n / mean_squared_norm) ** 0.5
activations_tensor = acts * scaling_factor

indices = torch.randperm(len(activations_tensor))
activations_tensor = activations_tensor[indices]

print("Saving dataset...")
torch.save(activations_tensor, "pile_activations.pt")
print(f"Dataset saved with {len(activations_tensor)} activation vectors.")
