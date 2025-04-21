import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
import os
import argparse

from sae import SAE

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


def train_sae(
    n_features,
    bandwidth,
    threshold,
    c,
    lambda_s_final,
    lambda_p,
    batch_size,
    learning_rate,
    epochs,
    grad_clip,
    data_path,
    save_path,
):
    wandb.init(
        project="sparse-autoencoder",
        config={
            "n_features": n_features,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lambda_s_final": lambda_s_final,
            "lambda_p": lambda_p,
        },
    )

    print("Loading data...")
    activations = torch.load(data_path)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() or 1,
    )

    model = SAE(
        d_model=activations.shape[1],
        n_features=n_features,
        bandwidth=bandwidth,
        threshold=threshold,
        lambda_p=lambda_p,
        c=c,
    ).to(device)

    init_sample = dataset[:10_000].unsqueeze(0).to(device)
    model.init_be(init_sample)

    model.compile()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
    )

    total_steps = epochs * len(dataloader)
    decay_steps = int(0.2 * total_steps)

    def lr_lambda(step):
        if step < (total_steps - decay_steps):
            return 1.0
        return max(0.0, 1.0 - (step - (total_steps - decay_steps)) / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("Starting training...")
    global_step = 0

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            x = batch.to(device)
            acts, x_hat = model(x)

            lambda_s = lambda_s_final * min(1.0, global_step / total_steps)

            loss = model.get_loss(x, x_hat, acts, lambda_s=lambda_s)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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

    torch.save(model.state_dict(), save_path)

    wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument("--n-features", type=int, default=18_432)
    parser.add_argument("--bandwidth", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=4.0, help="")
    parser.add_argument("--lambda-s-final", type=float, default=20.0)
    parser.add_argument("--lambda-p", type=float, default=3e-6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-path", type=str, default="sae_model.pt")
    parser.add_argument(
        "--data-path",
        type=str,
        default="pile_10k_activations.pt",
    )

    args = parser.parse_args()

    train_sae(
        n_features=args.n_features,
        bandwidth=args.bandwidth,
        threshold=args.threshold,
        c=args.c,
        lambda_s_final=args.lambda_s_final,
        lambda_p=args.lambda_p,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        data_path=args.data_path,
        save_path=args.save_path,
    )
