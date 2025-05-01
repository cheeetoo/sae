import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
import os
from datasets import load_dataset
import numpy as np

from sae import SAE

N_FEATURES = 18_432
BANDWIDTH = 2.0
THRESHOLD = 0.1
C = 4.0
LAMBDA_S_FINAL = 20.0
LAMBDA_P = 3e-6
BATCH_SIZE = 4096
LEARNING_RATE = 2e-4
EPOCHS = 10
GRAD_CLIP = 1.0
SAVE_PATH = "sae.pt"
REPO_ID = "cheeetoo/gemma-2-2b-fineweb-l13-acts"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

wandb.init(
    project="sparse-autoencoder",
    config={
        "n_features": N_FEATURES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lambda_s_final": LAMBDA_S_FINAL,
        "lambda_p": LAMBDA_P,
        "c": C,
        "bandwidth": BANDWIDTH,
        "threshold": THRESHOLD,
    },
)

print(f"Loading activations from {REPO_ID}")
hf_dataset = load_dataset(REPO_ID, split="train")
activations = torch.tensor(np.array(hf_dataset["activations"]), dtype=torch.float32)

dataset = TensorDataset(activations)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=os.cpu_count() or 1,
)

model = SAE(
    d_model=activations.shape[1],
    n_features=N_FEATURES,
    bandwidth=BANDWIDTH,
    threshold=THRESHOLD,
    lambda_p=LAMBDA_P,
    c=C,
).to(device)

init_sample = dataset[:10_000][0].unsqueeze(0).to(device)
model.init_be(init_sample)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

total_steps = EPOCHS * len(dataloader)
decay_steps = int(0.2 * total_steps)


def lr_lambda(step):
    if step < (total_steps - decay_steps):
        return 1.0
    return max(0.0, 1.0 - (step - (total_steps - decay_steps)) / decay_steps)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Starting training...")
global_step = 0

for epoch in range(EPOCHS):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in progress_bar:
        x = batch[0].to(device)
        acts, x_hat = model(x)

        lambda_s = LAMBDA_S_FINAL * min(1.0, global_step / total_steps)

        loss = model.get_loss(x, x_hat, acts, lambda_s=lambda_s)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            mse_loss = F.mse_loss(x_hat, x).item()
            sparsity = (acts > 0).float().mean().item()
            dead_features = ((acts.sum(dim=0) == 0).sum() / acts.shape[1]).item()

        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "mse": mse_loss,
                "sparsity": sparsity,
                "λ_S": lambda_s,
                "dead": f"{dead_features:.2%}",
            }
        )

        wandb.log(
            {
                "loss": loss.item(),
                "mse": mse_loss,
                "sparsity": sparsity,
                "lambda_s": lambda_s,
                "lr": scheduler.get_last_lr()[0],
                "dead_features": dead_features,
                "step": global_step,
                "epoch": epoch,
            }
        )

        global_step += 1

with torch.no_grad():
    wd_norm = torch.norm(model.w_d, dim=0, keepdim=True)

    model.w_e.data = model.w_e.data * wd_norm
    model.b_e = model.b_e * wd_norm.squeeze()
    model.w_d.data = model.w_d.data / wd_norm

torch.save(model.state_dict(), SAVE_PATH)

wandb.finish()

print("Training completed!")
