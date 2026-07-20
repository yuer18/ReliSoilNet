import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.utils.utils import TextColors as tc

# Setup device-agnostic code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimCLR(nn.Module):

    requires_aux = False

    def __init__(self, temperature: float):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be a positive float")
        self.temperature = temperature

    def forward(self, feats1, feats2=None):
        if isinstance(feats1, dict):
            feats2 = feats1["z_clim"]
            feats1 = feats1["z_img"]

        if feats2 is None:
            raise ValueError("SimCLR loss requires two feature tensors.")

        feats = torch.cat([feats1, feats2], dim=0)
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        acc_top1 = (sim_argsort == 0).float().mean()
        acc_top5 = (sim_argsort < 5).float().mean()
        acc_mean_pos = 1 + sim_argsort.float().mean()

        logs = {
            "contrastive_loss": float(nll.detach().item()),
            "img_mask_loss": 0.0,
            "clim_mask_loss": 0.0,
        }
        return nll, acc_top1, acc_top5, acc_mean_pos, logs


class HybridMultiModalSSLoss(nn.Module):


    requires_aux = True

    def __init__(
        self,
        temperature: float = 0.5,
        lambda_img_mask: float = 0.25,
        lambda_clim_mask: float = 1.0,
        climate_recon_loss: str = "smoothl1",
    ):
        super().__init__()
        self.contrastive = SimCLR(temperature=temperature)
        self.lambda_img_mask = float(lambda_img_mask)
        self.lambda_clim_mask = float(lambda_clim_mask)

        if climate_recon_loss == "smoothl1":
            self.climate_recon_loss = nn.SmoothL1Loss(reduction="mean")
        elif climate_recon_loss == "mse":
            self.climate_recon_loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError("climate_recon_loss must be one of ['smoothl1', 'mse']")

    def forward(self, model_outputs: Dict[str, torch.Tensor]):
        if not isinstance(model_outputs, dict):
            raise ValueError("HybridMultiModalSSLoss expects a dict of model outputs.")

        z_img = model_outputs["z_img"]
        z_clim = model_outputs["z_clim"]
        contrastive_loss, acc_top1, acc_top5, acc_mean_pos, _ = self.contrastive(z_img, z_clim)

        img_pred = F.normalize(model_outputs["img_recon"], dim=-1)
        img_tgt = F.normalize(model_outputs["img_target"], dim=-1)
        img_mask_loss = 1.0 - F.cosine_similarity(img_pred, img_tgt, dim=-1).mean()

        clim_recon = model_outputs["clim_recon"]
        clim_target = model_outputs["clim_target"]
        clim_time_mask = model_outputs["clim_time_mask"]
        clim_mask = clim_time_mask.unsqueeze(-1).expand_as(clim_target)

        if torch.any(clim_mask):
            clim_mask_loss = self.climate_recon_loss(clim_recon[clim_mask], clim_target[clim_mask])
        else:
            clim_mask_loss = torch.zeros((), device=clim_target.device)

        total_loss = (
            contrastive_loss
            + self.lambda_img_mask * img_mask_loss
            + self.lambda_clim_mask * clim_mask_loss
        )

        logs = {
            "contrastive_loss": float(contrastive_loss.detach().item()),
            "img_mask_loss": float(img_mask_loss.detach().item()),
            "clim_mask_loss": float(clim_mask_loss.detach().item()),
        }
        return total_loss, acc_top1, acc_top5, acc_mean_pos, logs


def test_SimCLR():
    from torch.distributions import uniform

    model = SimCLR(temperature=0.5)
    dummy_feats1 = uniform.Uniform(0, 1).rsample((64, 128))
    dummy_feats2 = uniform.Uniform(0, 1).rsample((64, 128))

    nll, acc_top1, acc_top5, acc_mean_pos, logs = model(dummy_feats1, dummy_feats2)

    assert isinstance(nll, torch.Tensor)
    assert isinstance(acc_top1, torch.Tensor)
    assert isinstance(acc_top5, torch.Tensor)
    assert isinstance(acc_mean_pos, torch.Tensor)
    assert isinstance(logs, dict)

    print(
        "nll:",
        nll,
        "\nacc_top1:",
        acc_top1,
        "\nacc_top5:",
        acc_top5,
        "\nacc_mean_pos:",
        acc_mean_pos,
        "\nlogs:",
        logs,
    )
    print("Test passed!")


def _move_inputs_to_device(X, y=None):
    if isinstance(X, (tuple, list)):
        X = [tensor.to(DEVICE) for tensor in list(X)]
    elif isinstance(X, torch.Tensor):
        X = X.to(DEVICE)
    else:
        raise ValueError(
            f"Input of the network must be either a Tensor or a Tuple/List of Tensors but it is: {type(X)}"
        )

    if y is not None:
        y = y.to(DEVICE)
        return X, y
    return X


def _forward_ssl_model(model: nn.Module, X, loss_fn: nn.Module):
    if getattr(loss_fn, "requires_aux", False):
        try:
            return model(X, return_aux=True)
        except TypeError:
            return model(X)
    return model(X)


def _compute_ssl_loss(loss_fn: nn.Module, model_outputs):
    if isinstance(model_outputs, dict):
        return loss_fn(model_outputs)
    if isinstance(model_outputs, (tuple, list)):
        return loss_fn(*model_outputs)
    return loss_fn(model_outputs)


def _accumulate_logs(sum_logs: Dict[str, float], batch_logs: Dict[str, float]):
    for k, v in batch_logs.items():
        sum_logs[k] = sum_logs.get(k, 0.0) + float(v)


# Train step function
def train_step(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss = 0.0
    train_top1 = 0.0
    train_top5 = 0.0
    train_mean_pos = 0.0
    train_logs_sum: Dict[str, float] = {}

    loop = tqdm(data_loader, leave=True)
    for batch, (X, y) in enumerate(loop):
        X, y = _move_inputs_to_device(X, y)

        model_outputs = _forward_ssl_model(model, X, loss_fn)
        loss, acc_top1, acc_top5, acc_mean_pos, batch_logs = _compute_ssl_loss(loss_fn, model_outputs)

        train_loss += float(loss.item())
        train_top1 += float(acc_top1.item())
        train_top5 += float(acc_top5.item())
        train_mean_pos += float(acc_mean_pos.item())
        _accumulate_logs(train_logs_sum, batch_logs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(data_loader) - 1:
            postfix = {"Train_Loss": train_loss / (batch + 1)}
            if "contrastive_loss" in train_logs_sum:
                postfix["Ctr"] = train_logs_sum["contrastive_loss"] / (batch + 1)
            if train_logs_sum.get("img_mask_loss", 0.0) > 0:
                postfix["Img"] = train_logs_sum["img_mask_loss"] / (batch + 1)
            if train_logs_sum.get("clim_mask_loss", 0.0) > 0:
                postfix["Clim"] = train_logs_sum["clim_mask_loss"] / (batch + 1)
            loop.set_postfix(**postfix)

    num_batches = max(len(data_loader), 1)
    train_loss /= num_batches
    train_top1 /= num_batches
    train_top5 /= num_batches
    train_mean_pos /= num_batches
    train_logs = {k: v / num_batches for k, v in train_logs_sum.items()}

    return train_loss, train_top1, train_top5, train_mean_pos, train_logs


# Validation step function
def test_step(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, verbose: bool = False):
    model.eval()
    test_loss = 0.0
    test_top1 = 0.0
    test_top5 = 0.0
    test_mean_pos = 0.0
    test_logs_sum: Dict[str, float] = {}
    last_outputs = None

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = _move_inputs_to_device(X, y)
            model_outputs = _forward_ssl_model(model, X, loss_fn)
            loss, acc_top1, acc_top5, acc_mean_pos, batch_logs = _compute_ssl_loss(loss_fn, model_outputs)

            test_loss += float(loss.item())
            test_top1 += float(acc_top1.item())
            test_top5 += float(acc_top5.item())
            test_mean_pos += float(acc_mean_pos.item())
            _accumulate_logs(test_logs_sum, batch_logs)
            last_outputs = model_outputs

    num_batches = max(len(data_loader), 1)
    test_loss /= num_batches
    test_top1 /= num_batches
    test_top5 /= num_batches
    test_mean_pos /= num_batches
    test_logs = {k: v / num_batches for k, v in test_logs_sum.items()}

    if verbose and last_outputs is not None:
        print(f"Test Loss: {test_loss:>8f}%")
        if isinstance(last_outputs, dict):
            print(last_outputs["z_img"].shape, last_outputs["z_clim"].shape)
        else:
            print(last_outputs[0].shape, last_outputs[1].shape)

    return test_loss, test_top1, test_top5, test_mean_pos, test_logs


def save_checkpoint(model, optimizer, filename: str = "my_checkpoint.pth.tar"):
    print("Saving checkpoint=> ", end="")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print("Done!")


def load_checkpoint(model, optimizer, filename: str = "my_checkpoint.pth.tar"):
    print("Loading checkpoint=> ", end="")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Done!")


# Training loop for SSL pretraining
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = SimCLR(temperature=0.5),
    epochs: int = 5,
    lr_scheduler: Optional[str] = None,
    save_model_path: Optional[str] = None,
    save_model_if_mae_lower_than=None,
):
    _ = test_dataloader
    _ = save_model_path
    _ = save_model_if_mae_lower_than

    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    else:
        scheduler = None

    results = {
        "train_loss": [],
        "train_acc_top1": [],
        "train_acc_top5": [],
        "train_acc_mean_pos": [],
        "val_loss": [],
        "val_acc_top1": [],
        "val_acc_top5": [],
        "val_acc_mean_pos": [],
        "train_loss_components": [],
        "val_loss_components": [],
    }

    for epoch in range(1, epochs + 1):
        print(tc.OKGREEN, f"Epoch {epoch}\n-------------------------------", tc.ENDC)

        train_loss, train_acc_top1, train_acc_top5, train_acc_mean_pos, train_logs = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc_top1, val_acc_top5, val_acc_mean_pos, val_logs = test_step(
            model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
        )

        print(
            tc.OKCYAN,
            f"Epoch {epoch} Results: | ",
            f"train_loss: {train_loss:.6f} | ",
            f"val_loss: {val_loss:.6f} | ",
            f"train_acc_top1: {train_acc_top1:.6f} | ",
            f"val_acc_top1: {val_acc_top1:.6f} | ",
            f"train_acc_top5: {train_acc_top5:.6f} | ",
            f"val_acc_top5: {val_acc_top5:.6f} | ",
            f"train_acc_mean_pos: {train_acc_mean_pos:.6f} | ",
            f"val_acc_mean_pos: {val_acc_mean_pos:.6f}",
            tc.ENDC,
        )
        if train_logs or val_logs:
            print(
                tc.OKBLUE,
                f"Loss breakdown | train: {train_logs} | val: {val_logs}",
                tc.ENDC,
            )
        print("")

        results["train_loss"].append(train_loss)
        results["train_acc_top1"].append(train_acc_top1)
        results["train_acc_top5"].append(train_acc_top5)
        results["train_acc_mean_pos"].append(train_acc_mean_pos)
        results["val_loss"].append(val_loss)
        results["val_acc_top1"].append(val_acc_top1)
        results["val_acc_top5"].append(val_acc_top5)
        results["val_acc_mean_pos"].append(val_acc_mean_pos)
        results["train_loss_components"].append(train_logs)
        results["val_loss_components"].append(val_logs)

        if scheduler is not None:
            if lr_scheduler == "step":
                scheduler.step()
            elif lr_scheduler == "plateau":
                scheduler.step(train_loss)

    return results


# Simple plotting helper
def plot_losses(loss_dict):
    train_losses = loss_dict["train_loss"]
    val_losses = loss_dict["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class BatchLoader(torch.utils.data.Dataset):
    """Takes in a PyTorch DataLoader and returns any batch by index."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __call__(self, index):
        for i, batch in enumerate(self.dataloader):
            if i == index:
                return batch
        raise IndexError("Index out of range")


if __name__ == "__main__":
    test_SimCLR()
