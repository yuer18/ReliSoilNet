import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.utils.utils import TextColors as tc

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class R2Loss(nn.Module):
    """
    Calculates the R2 loss for regression problems.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        ones = torch.ones_like(y)
        return 1 - (self.mse(yhat, y) / self.mse(y, ones * y.mean()))


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, actuals):

        predictions = torch.clamp(predictions, min=-1 + 1e-9)
        actuals = torch.clamp(actuals, min=-1 + 1e-9)
        log_diff = torch.log(predictions + 1) - torch.log(actuals + 1)
        squared_log_diff = torch.square(log_diff)
        return torch.sqrt(torch.mean(squared_log_diff))


def _move_inputs_to_device(X, y=None):
    if isinstance(X, (tuple, list)):
        X = [tensor.to(device) for tensor in list(X)]
    elif isinstance(X, torch.Tensor):
        X = X.to(device)
    else:
        raise ValueError(
            f"Input of the network must be either a Tensor or a Tuple/List of Tensors but it is: {type(X)}"
        )

    if y is not None:
        y = y.to(device)
        return X, y
    return X


class _RobustMode:
    NONE = "none"
    FEATURE_FALLBACK = "feature_fallback"


class _GateLossMode:
    MARGIN = "margin"


_AUX_EVAL_MODULE_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@contextmanager
def _temporary_eval_modules(module: nn.Module, module_types=_AUX_EVAL_MODULE_TYPES):
    toggled_modules = []
    for submodule in module.modules():
        if isinstance(submodule, module_types) and submodule.training:
            submodule.eval()
            toggled_modules.append(submodule)
    try:
        yield
    finally:
        for submodule in toggled_modules:
            submodule.train()


def _sanitize_robust_training_config(robust_training_config: Optional[Dict]) -> Optional[Dict]:
    if robust_training_config is None:
        return None

    clean = {}
    for k, v in robust_training_config.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            continue
        clean[k] = v
    return clean


def _supports_feature_fallback(model: nn.Module, X, robust_training_config: Optional[Dict]) -> bool:
    if robust_training_config is None:
        return False
    if robust_training_config.get("mode", _RobustMode.NONE) != _RobustMode.FEATURE_FALLBACK:
        return False
    if not isinstance(X, list) or len(X) < 2:
        return False
    if not hasattr(model, "soilnet_simclr") or not hasattr(model, "reg"):
        return False
    return True


def _empty_degradation_meta(batch_size: int, ref_device: torch.device) -> Dict:
    return {
        "batch_applied": False,
        "degraded_any": False,
        "affected_mask": torch.zeros(batch_size, dtype=torch.bool, device=ref_device),
        "img_sample_mask": torch.zeros(batch_size, dtype=torch.bool, device=ref_device),
        "clim_sample_mask": torch.zeros(batch_size, dtype=torch.bool, device=ref_device),
        "local_img_sample_mask": torch.zeros(0, dtype=torch.bool, device=ref_device),
        "local_clim_sample_mask": torch.zeros(0, dtype=torch.bool, device=ref_device),
        "affected_count": 0,
        "selected_batch_size": 0,
        "subset_ratio": 0.0,
    }


def _sample_modality_masks(batch_size: int, ref_device: torch.device, robust_training_config: Dict) -> Dict:
    batch_prob = float(robust_training_config.get("batch_prob", 1.0))
    batch_prob = max(0.0, min(1.0, batch_prob))
    if batch_prob <= 0.0 or float(torch.rand(1, device=ref_device).item()) >= batch_prob:
        return _empty_degradation_meta(batch_size, ref_device)

    drop_prob = float(robust_training_config.get("drop_prob", 0.0))
    drop_prob = max(0.0, min(1.0, drop_prob))
    if drop_prob <= 0.0:
        meta = _empty_degradation_meta(batch_size, ref_device)
        meta["batch_applied"] = True
        return meta

    modality = robust_training_config.get("modality", "climate")
    base_sample_mask = torch.rand(batch_size, device=ref_device) < drop_prob

    img_sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=ref_device)
    clim_sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=ref_device)

    if modality == "image":
        img_sample_mask = base_sample_mask
    elif modality == "climate":
        clim_sample_mask = base_sample_mask
    elif modality == "both":
        img_sample_mask = base_sample_mask
        clim_sample_mask = base_sample_mask
    elif modality == "random_one":
        route_mask = torch.rand(batch_size, device=ref_device) < 0.5
        img_sample_mask = base_sample_mask & route_mask
        clim_sample_mask = base_sample_mask & (~route_mask)
    else:
        raise ValueError("robust_modality must be one of ['random_one', 'image', 'climate', 'both']")

    affected_mask = img_sample_mask | clim_sample_mask
    affected_count = int(affected_mask.sum().item())
    min_samples = max(1, int(robust_training_config.get("min_samples", 1)))
    if affected_count < min_samples:
        meta = _empty_degradation_meta(batch_size, ref_device)
        meta["batch_applied"] = True
        return meta

    return {
        "batch_applied": True,
        "degraded_any": True,
        "affected_mask": affected_mask,
        "img_sample_mask": img_sample_mask,
        "clim_sample_mask": clim_sample_mask,
        "local_img_sample_mask": img_sample_mask[affected_mask],
        "local_clim_sample_mask": clim_sample_mask[affected_mask],
        "affected_count": affected_count,
        "selected_batch_size": affected_count,
        "subset_ratio": float(affected_count / max(batch_size, 1)),
    }


def _mean_feature_prototype(features: torch.Tensor, reference_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if reference_mask is not None and torch.any(reference_mask):
        return features[reference_mask].detach().mean(dim=0, keepdim=True)
    return features.detach().mean(dim=0, keepdim=True)


def _update_running_prototype(state: Dict, key: str, batch_proto: torch.Tensor, momentum: float) -> torch.Tensor:
    if key not in state or state[key] is None:
        state[key] = batch_proto.detach().clone()
    else:
        state[key] = momentum * state[key] + (1.0 - momentum) * batch_proto.detach()
    return state[key]


def _get_feature_prototypes(
    img_feat: torch.Tensor,
    clim_feat: torch.Tensor,
    degradation_meta: Dict,
    robust_training_config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state = robust_training_config.setdefault("_state", {})
    momentum = float(robust_training_config.get("prototype_momentum", 0.90))
    momentum = max(0.0, min(0.9999, momentum))

    img_reference_mask = ~degradation_meta["img_sample_mask"]
    clim_reference_mask = ~degradation_meta["clim_sample_mask"]

    batch_img_proto = _mean_feature_prototype(img_feat, img_reference_mask)
    batch_clim_proto = _mean_feature_prototype(clim_feat, clim_reference_mask)

    img_proto = _update_running_prototype(state, "img_proto", batch_img_proto, momentum)
    clim_proto = _update_running_prototype(state, "clim_proto", batch_clim_proto, momentum)
    return img_proto, clim_proto


def _degrade_feature_subset(
    img_subset: torch.Tensor,
    clim_subset: torch.Tensor,
    local_img_sample_mask: torch.Tensor,
    local_clim_sample_mask: torch.Tensor,
    img_proto: torch.Tensor,
    clim_proto: torch.Tensor,
    robust_training_config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    degraded_img = img_subset.clone()
    degraded_clim = clim_subset.clone()

    img_severity = float(robust_training_config.get("img_severity", 0.30))
    clim_severity = float(robust_training_config.get("clim_severity", 0.40))
    img_severity = max(0.0, min(1.0, img_severity))
    clim_severity = max(0.0, min(1.0, clim_severity))

    if torch.any(local_img_sample_mask):
        degraded_img[local_img_sample_mask] = (
            (1.0 - img_severity) * degraded_img[local_img_sample_mask]
            + img_severity * img_proto.expand(int(local_img_sample_mask.sum().item()), -1)
        )

    if torch.any(local_clim_sample_mask):
        degraded_clim[local_clim_sample_mask] = (
            (1.0 - clim_severity) * degraded_clim[local_clim_sample_mask]
            + clim_severity * clim_proto.expand(int(local_clim_sample_mask.sum().item()), -1)
        )

    return degraded_img, degraded_clim


def _register_gate_logits_hook(model: nn.Module):
    captured: Dict[str, torch.Tensor] = {}
    gate_net = getattr(getattr(model, "reg", None), "gate_net", None)
    if gate_net is None or not isinstance(gate_net, nn.Module):
        return captured, None

    def _hook(_module, _inputs, output):
        captured["gate_logits"] = output

    handle = gate_net.register_forward_hook(_hook)
    return captured, handle


def _compute_gate_margin_loss(
    gate_logits: Optional[torch.Tensor],
    local_img_sample_mask: torch.Tensor,
    local_clim_sample_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if gate_logits is None:
        return torch.zeros((), device=local_img_sample_mask.device)

    gate_weights = torch.softmax(gate_logits, dim=1)
    img_gate = gate_weights[:, 0]
    clim_gate = gate_weights[:, 1]

    losses = []

    climate_only = local_clim_sample_mask & (~local_img_sample_mask)
    if torch.any(climate_only):
        losses.append(F.relu(margin - (img_gate[climate_only] - clim_gate[climate_only])).mean())

    image_only = local_img_sample_mask & (~local_clim_sample_mask)
    if torch.any(image_only):
        losses.append(F.relu(margin - (clim_gate[image_only] - img_gate[image_only])).mean())

    if len(losses) == 0:
        return torch.zeros((), device=local_img_sample_mask.device)

    return torch.stack(losses).mean()


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    robust_training_config: Optional[Dict] = None,
):
    model.train()
    train_loss = 0.0
    train_logs_sum = {
        "base_supervised_loss": 0.0,
        "degraded_supervised_loss": 0.0,
        "consistency_loss": 0.0,
        "gate_margin_loss": 0.0,
        "affected_ratio": 0.0,
        "image_affected_ratio": 0.0,
        "climate_affected_ratio": 0.0,
        "robust_batch_apply_ratio": 0.0,
        "robust_selected_subset_ratio": 0.0,
    }
    robust_enabled = robust_training_config is not None and robust_training_config.get("mode", _RobustMode.NONE) != _RobustMode.NONE

    loop = tqdm(data_loader, leave=True)

    for batch, (X, y) in enumerate(loop):
        X, y = _move_inputs_to_device(X, y)
        y_target = y.unsqueeze(1)

        feature_fallback_supported = _supports_feature_fallback(model, X, robust_training_config)
        if feature_fallback_supported:
            img_feat, clim_feat = model.soilnet_simclr((X[0], X[1]), return_aux=False)
            y_pred_full = model.reg(img_feat, clim_feat)
        else:
            y_pred_full = model(X)
            img_feat, clim_feat = None, None

        base_supervised_loss = loss_fn(y_pred_full, y_target)
        loss = base_supervised_loss

        degraded_supervised_loss = torch.zeros((), device=y_target.device)
        consistency_loss = torch.zeros((), device=y_target.device)
        gate_margin_loss = torch.zeros((), device=y_target.device)

        if robust_enabled and feature_fallback_supported:
            degradation_meta = _sample_modality_masks(X[0].shape[0], X[0].device, robust_training_config)
            if degradation_meta["batch_applied"]:
                train_logs_sum["robust_batch_apply_ratio"] += 1.0

            if degradation_meta["degraded_any"]:
                affected_mask = degradation_meta["affected_mask"]
                local_img_sample_mask = degradation_meta["local_img_sample_mask"]
                local_clim_sample_mask = degradation_meta["local_clim_sample_mask"]
                subset_ratio = float(degradation_meta["subset_ratio"])

                img_proto, clim_proto = _get_feature_prototypes(
                    img_feat=img_feat,
                    clim_feat=clim_feat,
                    degradation_meta=degradation_meta,
                    robust_training_config=robust_training_config,
                )

                img_subset = img_feat[affected_mask]
                clim_subset = clim_feat[affected_mask]
                if bool(robust_training_config.get("stopgrad_aux", True)):
                    img_subset = img_subset.detach()
                    clim_subset = clim_subset.detach()

                degraded_img_subset, degraded_clim_subset = _degrade_feature_subset(
                    img_subset=img_subset,
                    clim_subset=clim_subset,
                    local_img_sample_mask=local_img_sample_mask,
                    local_clim_sample_mask=local_clim_sample_mask,
                    img_proto=img_proto,
                    clim_proto=clim_proto,
                    robust_training_config=robust_training_config,
                )

                gate_capture, gate_hook = _register_gate_logits_hook(model)
                try:
                    if bool(robust_training_config.get("freeze_aux_stochastic", True)):
                        with _temporary_eval_modules(model.reg):
                            y_pred_degraded = model.reg(degraded_img_subset, degraded_clim_subset)
                    else:
                        y_pred_degraded = model.reg(degraded_img_subset, degraded_clim_subset)
                finally:
                    if gate_hook is not None:
                        gate_hook.remove()

                teacher_blend = float(robust_training_config.get("teacher_blend", 0.25))
                teacher_blend = max(0.0, min(1.0, teacher_blend))
                soft_teacher_target = (
                    teacher_blend * y_target[affected_mask]
                    + (1.0 - teacher_blend) * y_pred_full.detach()[affected_mask]
                )
                degraded_supervised_loss = F.smooth_l1_loss(y_pred_degraded, soft_teacher_target)
                consistency_loss = F.smooth_l1_loss(y_pred_degraded, y_pred_full.detach()[affected_mask])

                gate_logits = gate_capture.get("gate_logits", None)
                gate_margin = float(robust_training_config.get("gate_margin", 0.10))
                gate_margin_loss = _compute_gate_margin_loss(
                    gate_logits=gate_logits,
                    local_img_sample_mask=local_img_sample_mask,
                    local_clim_sample_mask=local_clim_sample_mask,
                    margin=gate_margin,
                )

                loss = (
                    base_supervised_loss
                    + float(robust_training_config.get("supervised_weight", 0.12)) * degraded_supervised_loss
                    + float(robust_training_config.get("consistency_weight", 0.05)) * consistency_loss
                    + float(robust_training_config.get("gate_weight", 0.08)) * gate_margin_loss
                )

                train_logs_sum["degraded_supervised_loss"] += float(degraded_supervised_loss.item())
                train_logs_sum["consistency_loss"] += float(consistency_loss.item())
                train_logs_sum["gate_margin_loss"] += float(gate_margin_loss.item())
                train_logs_sum["affected_ratio"] += float(affected_mask.float().mean().item())
                train_logs_sum["image_affected_ratio"] += float(
                    degradation_meta["img_sample_mask"].float().mean().item()
                )
                train_logs_sum["climate_affected_ratio"] += float(
                    degradation_meta["clim_sample_mask"].float().mean().item()
                )
                train_logs_sum["robust_selected_subset_ratio"] += subset_ratio

        train_logs_sum["base_supervised_loss"] += float(base_supervised_loss.item())
        train_loss += float(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(data_loader) - 1:
            postfix = {"Train_Loss": train_loss / (batch + 1)}
            if robust_enabled:
                postfix["Base"] = train_logs_sum["base_supervised_loss"] / (batch + 1)
                postfix["Aux"] = train_logs_sum["degraded_supervised_loss"] / (batch + 1)
                postfix["Cons"] = train_logs_sum["consistency_loss"] / (batch + 1)
                postfix["Gate"] = train_logs_sum["gate_margin_loss"] / (batch + 1)
                postfix["RB"] = train_logs_sum["robust_batch_apply_ratio"] / (batch + 1)
                postfix["RS"] = train_logs_sum["robust_selected_subset_ratio"] / (batch + 1)
            loop.set_postfix(**postfix)

    num_batches = max(len(data_loader), 1)
    train_loss = train_loss / num_batches
    train_logs = {k: v / num_batches for k, v in train_logs_sum.items()}
    return train_loss, train_logs


def test_step(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, verbose: bool = False):
    model.eval()
    test_loss = 0.0
    last_y_pred = None
    last_y = None

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = _move_inputs_to_device(X, y)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += float(loss.item())
            last_y_pred = y_pred
            last_y = y

    test_loss /= len(data_loader)

    if verbose and last_y_pred is not None and last_y is not None:
        print(f"Test Loss: {test_loss:>8f}%")
        print(last_y_pred.shape, last_y.shape)

    return test_loss


def predict_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    denorm_factor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = _move_inputs_to_device(X, y)
            y_pred = model(X)

            y_true_np = y.detach().cpu().numpy().reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

            y_true_all.append(y_true_np)
            y_pred_all.append(y_pred_np)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([], dtype=np.float32)
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([], dtype=np.float32)

    if denorm_factor is not None:
        y_true = y_true * denorm_factor
        y_pred = y_pred * denorm_factor

    return y_true, y_pred


def predict_on_loader_w_id(
    model: nn.Module,
    data_loader: DataLoader,
    denorm_factor: Optional[float] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    point_ids: List[str] = []
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    with torch.inference_mode():
        for X, y, batch_point_ids in data_loader:
            X, y = _move_inputs_to_device(X, y)
            y_pred = model(X)

            y_true_np = y.detach().cpu().numpy().reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

            point_ids.extend([str(pid) for pid in batch_point_ids])
            y_true_all.append(y_true_np)
            y_pred_all.append(y_pred_np)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([], dtype=np.float32)
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([], dtype=np.float32)

    if denorm_factor is not None:
        y_true = y_true * denorm_factor
        y_pred = y_pred * denorm_factor

    return point_ids, y_true, y_pred


def test_step_w_id(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    csv_file: str = "test.csv",
    verbose: bool = False,
):
    model.eval()
    test_loss = 0.0
    results = []
    last_y_pred = None
    last_y = None

    with torch.inference_mode():
        for X, y, point_id in data_loader:
            X, y = _move_inputs_to_device(X, y)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += float(loss.item())
            last_y_pred = y_pred
            last_y = y

            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
            y_true_np = y.detach().cpu().numpy().reshape(-1)

            for i in range(len(point_id)):
                results.append(
                    {
                        "point_id": str(point_id[i]),
                        "y_real": float(y_true_np[i]),
                        "y_pred": float(y_pred_np[i]),
                    }
                )

    test_loss /= len(data_loader)

    if verbose and last_y_pred is not None and last_y is not None:
        print(f"Test Loss: {test_loss:>8f}%")
        print(last_y_pred.shape, last_y.shape)

    df = pd.DataFrame(results)
    if csv_file:
        df.to_csv(csv_file, index=False)

    return test_loss, df


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filename: str = "my_checkpoint.pth.tar",
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
):
    print("Saving checkpoint=> ", end="")
    checkpoint = {
        "state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, filename)
    print("Done!")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filename: str = "my_checkpoint.pth.tar",
):
    print("Loading checkpoint=> ", end="")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Done!")
    return checkpoint



def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: Optional[torch.utils.data.DataLoader],
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = RMSELoss(),
    epochs: int = 5,
    lr_scheduler: Optional[str] = None,
    save_model_path: Optional[str] = None,
    save_model_if_mae_lower_than=None,
    save_train_data_metrics: bool = False,
    oc_max: float = 1.0,
    robust_training_config: Optional[Dict] = None,
):

    _ = test_dataloader
    _ = save_model_if_mae_lower_than

    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    else:
        scheduler = None

    robust_training_config = dict(robust_training_config) if robust_training_config is not None else None
    if robust_training_config is not None:
        robust_training_config["_state"] = {
            "img_proto": None,
            "clim_proto": None,
        }

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_aux_metrics_per_epoch": [],
        "val_metrics_per_epoch": [],
        "best_epoch": None,
        "best_val_metrics": None,
        "final_val_metrics": None,
        "train_metrics": None,
        "robust_training_config": _sanitize_robust_training_config(robust_training_config),
    }

    best_val_rmse = float("inf")
    best_val_ccc = float("-inf")
    robust_mode = _RobustMode.NONE if robust_training_config is None else robust_training_config.get("mode", _RobustMode.NONE)
    robust_warmup_epochs = 0 if robust_training_config is None else int(robust_training_config.get("warmup_epochs", 0))

    robust_supported = robust_mode == _RobustMode.NONE or (
        hasattr(model, "soilnet_simclr") and hasattr(model, "reg")
    )
    if robust_mode != _RobustMode.NONE and not robust_supported:
        print(
            tc.WARNING,
            "Robust feature-fallback training was requested, but the current model does not expose a multimodal SSL encoder + fusion head. "
            "Training will continue with the original fine-tuning objective.",
            tc.ENDC,
        )

    for epoch in range(1, epochs + 1):
        print(tc.OKGREEN, f"Epoch {epoch}\n-------------------------------", tc.ENDC)

        active_robust_config = robust_training_config
        if robust_mode == _RobustMode.NONE or not robust_supported:
            active_robust_config = None
        elif epoch <= robust_warmup_epochs:
            active_robust_config = None
            print(
                tc.WARNING,
                f"Robust multimodal learning warmup active ({epoch}/{robust_warmup_epochs}). This epoch uses the original full-modality training objective. The feature-fallback auxiliary branch will start after warmup.",
                tc.ENDC,
            )

        train_loss, train_aux_logs = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            robust_training_config=active_robust_config,
        )
        val_loss = test_step(model=model, data_loader=val_dataloader, loss_fn=loss_fn)

        y_true_val, y_pred_val = predict_on_loader(model, val_dataloader, denorm_factor=oc_max)
        val_rmse, val_r2, val_rpiq, val_mae, val_mec, val_ccc = evaluate_regression_metrics(
            y_true_val, y_pred_val
        )

        current_val_metrics = {
            "epoch": int(epoch),
            "MAE": float(val_mae),
            "RMSE": float(val_rmse),
            "R2": float(val_r2),
            "RPIQ": float(val_rpiq),
            "MEC": float(val_mec),
            "CCC": float(val_ccc),
        }

        print(
            tc.OKCYAN,
            f"Epoch {epoch} Results: | ",
            f"train_loss: {train_loss:.6f} | ",
            f"val_loss: {val_loss:.6f} | ",
            f"val_MAE(real-scale): {val_mae:.6f} | ",
            f"val_RMSE(real-scale): {val_rmse:.6f} | ",
            f"val_R2(real-scale): {val_r2:.6f} | ",
            f"val_RPIQ(real-scale): {val_rpiq:.6f} | ",
            f"val_CCC(real-scale): {val_ccc:.6f}",
            tc.ENDC,
        )

        epoch_train_aux_logs = None
        if robust_mode != _RobustMode.NONE:
            epoch_train_aux_logs = dict(train_aux_logs) if train_aux_logs is not None else {}
            epoch_train_aux_logs["robust_mode"] = robust_mode
            epoch_train_aux_logs["robust_active"] = bool(active_robust_config is not None)
            epoch_train_aux_logs["robust_warmup_active"] = bool(
                robust_mode != _RobustMode.NONE and epoch <= robust_warmup_epochs
            )

            print(
                tc.OKBLUE,
                "Train breakdown | ",
                f"base_sup: {epoch_train_aux_logs['base_supervised_loss']:.6f} | ",
                f"robust_aux: {epoch_train_aux_logs['degraded_supervised_loss']:.6f} | ",
                f"cons: {epoch_train_aux_logs['consistency_loss']:.6f} | ",
                f"gate: {epoch_train_aux_logs['gate_margin_loss']:.6f} | ",
                f"affected: {epoch_train_aux_logs['affected_ratio']:.4f} | ",
                f"img_aff: {epoch_train_aux_logs['image_affected_ratio']:.4f} | ",
                f"clim_aff: {epoch_train_aux_logs['climate_affected_ratio']:.4f} | ",
                f"subset: {epoch_train_aux_logs['robust_selected_subset_ratio']:.4f}",
                tc.ENDC,
            )
        print("")

        results["train_loss"].append(float(train_loss))
        results["val_loss"].append(float(val_loss))
        results["train_aux_metrics_per_epoch"].append(epoch_train_aux_logs)
        results["val_metrics_per_epoch"].append(current_val_metrics)
        results["final_val_metrics"] = current_val_metrics

        is_better = (val_rmse < best_val_rmse) or (
            np.isclose(val_rmse, best_val_rmse) and val_ccc > best_val_ccc
        )

        if is_better:
            best_val_rmse = float(val_rmse)
            best_val_ccc = float(val_ccc)
            results["best_epoch"] = int(epoch)
            results["best_val_metrics"] = current_val_metrics

            if save_model_path is not None:
                save_checkpoint(
                    model,
                    optimizer=optimizer,
                    filename=save_model_path,
                    epoch=epoch,
                    metrics=current_val_metrics,
                )

        if scheduler is not None:
            if lr_scheduler == "step":
                scheduler.step()
            elif lr_scheduler == "plateau":
                scheduler.step(val_loss)

    if save_model_path is not None and os.path.exists(save_model_path):
        load_checkpoint(model, optimizer=None, filename=save_model_path)

    if save_train_data_metrics:
        y_true_train, y_pred_train = predict_on_loader(model, train_dataloader, denorm_factor=oc_max)
        train_rmse, train_r2, train_rpiq, train_mae, train_mec, train_ccc = evaluate_regression_metrics(
            y_true_train, y_pred_train
        )
        results["train_metrics"] = {
            "MAE": float(train_mae),
            "RMSE": float(train_rmse),
            "R2": float(train_r2),
            "RPIQ": float(train_rpiq),
            "MEC": float(train_mec),
            "CCC": float(train_ccc),
        }

    return results


def plot_losses(loss_dict):
    train_losses = loss_dict["train_loss"]
    val_losses = loss_dict["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class BatchLoader(torch.utils.data.Dataset):


    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __call__(self, index):
        for i, batch in enumerate(self.dataloader):
            if i == index:
                return batch
        raise IndexError("Index out of range")


def evaluate_regression_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        raise ValueError("No valid samples found when computing regression metrics.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    mec = np.mean(y_true - y_pred)

    if y_true.size < 2:
        r2 = float("nan")
    else:
        r2 = r2_score(y_true, y_pred)

    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    if rmse == 0:
        rpiq = float("inf")
    else:
        rpiq = (q3 - q1) / rmse

    if y_true.size < 2:
        ccc = float("nan")
    else:
        cor = np.corrcoef(y_true, y_pred)[0, 1]
        mean_real = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_real = np.var(y_true)
        var_pred = np.var(y_pred)
        sd_real = np.std(y_true)
        sd_pred = np.std(y_pred)
        denominator = var_real + var_pred + (mean_real - mean_pred) ** 2
        ccc = float("nan") if denominator == 0 else (2 * cor * sd_real * sd_pred) / denominator

    return rmse, r2, rpiq, mae, mec, ccc



class PhysicsPinballLoss(nn.Module):

    def __init__(self, q, beta):
        super().__init__()
        self.q = q
        self.beta = beta

    def forward(self, y_pred, y_true):
        if self.q >= 0.5:
            raise ValueError("The input quantile should be lower than 0.5")

        e = y_true - y_pred
        loss_lower = torch.mean(torch.max(self.q * e, (self.q - 1) * e))
        loss_upper = torch.mean(torch.max((1 - self.q) * e, ((1 - self.q) - 1) * e))

        lower_bound = y_pred - loss_upper
        upper_bound = y_pred - loss_lower

        penalty_lower = torch.where(
            y_true < lower_bound,
            self.beta * (lower_bound - y_true),
            torch.tensor(0.0, device=device),
        )
        penalty_upper = torch.where(
            y_true > upper_bound,
            self.beta * (y_true - upper_bound),
            torch.tensor(0.0, device=device),
        )

        loss = loss_lower + loss_upper + torch.mean(penalty_lower) + torch.mean(penalty_upper)
        return loss
