import argparse
import json
import os
import random
import shutil
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from soilnet.soil_net import SoilNetSimCLR
from train_SimCLR_utils import HybridMultiModalSSLoss, SimCLR, train


# create a folder called 'results' in the current directory if it doesn't exist
if not os.path.exists("results"):
    os.mkdir("results")

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
EXP_NAME = "LUCAS_Self560_ViT_Trans"
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
LR_SCHEDULER = "step"  # step, plateau or None
DATASET = "LUCAS"  # 'LUCAS', 'RaCA'
USE_SRTM = False
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = "ViT"  # vgg16 or resnet101 or "ViT"
RNN_ARCHITECTURE = "Transformer"  # LSTM, GRU, RNN, Transformer
REG_VERSION = 1
USE_LSTM_BRANCH = False
SEEDS = [1]
SSL_OBJECTIVE = "contrastive"  # contrastive or hybrid
TEMPERATURE = 0.5
LAMBDA_IMG_MASK = 0.25
LAMBDA_CLIM_MASK = 1.0
IMG_MASK_RATIO = 0.35
CLIM_MASK_RATIO = 0.25
CLIM_MASK_MODE = "block"
CLIM_BLOCK_SIZE = 3
IMG_MASK_PATCH_SIZE = 8
IMG_DECODER_HIDDEN = 256
CLIM_DECODER_HIDDEN = 256
AUX_DROPOUT = 0.1


train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path


DEFAULT_OC_MAX = {
    "LUCAS": 560.2,
    "RaCA": 4115.0,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="SoilNet SSL Training")
    parser.add_argument("-exp", "--experiment_name", type=str, default=EXP_NAME, help="Experiment name")
    parser.add_argument("-nw", "--num_workers", type=int, default=NUM_WORKERS, help="Number of workers for data loading")
    parser.add_argument("-trbs", "--train_batch_size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("-tsbs", "--test_batch_size", type=int, default=TEST_BATCH_SIZE, help="Batch size for testing")
    parser.add_argument("-lr", "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("-ne", "--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("-lrs", "--lr_scheduler", type=str, default=LR_SCHEDULER, choices=["step", "plateau", "None"], help="Learning rate scheduler")
    parser.add_argument("-ds", "--dataset", type=str, default=DATASET, choices=["LUCAS", "RaCA"], help="Dataset name")
    parser.add_argument("-srtm", "--use_srtm", action="store_true", default=USE_SRTM, help="Use SRTM data")
    parser.add_argument("-sa", "--use_spatial_attention", action="store_true", default=USE_SPATIAL_ATTENTION, help="Use spatial attention")
    parser.add_argument("-cnn", "--cnn_architecture", type=str, default=CNN_ARCHITECTURE, choices=["vgg16", "resnet101", "ViT", "resnet50"], help="CNN architecture")
    parser.add_argument("-rnn", "--rnn_architecture", type=str, default=RNN_ARCHITECTURE, choices=["LSTM", "GRU", "RNN", "Transformer"], help="RNN architecture")
    parser.add_argument("-rv", "--reg_version", type=int, default=REG_VERSION, choices=[1, 2], help="Regression version")
    parser.add_argument("-lstm", "--use_lstm_branch", action="store_true", default=USE_LSTM_BRANCH, help="Use climate data branch")
    parser.add_argument("-s", "--seeds", nargs="+", type=int, default=SEEDS, help="Seeds for cross validation")

    parser.add_argument("--ssl_objective", type=str, default=SSL_OBJECTIVE, choices=["contrastive", "hybrid"], help="SSL objective: original contrastive or proposed hybrid objective")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="InfoNCE temperature")
    parser.add_argument("--lambda_img_mask", type=float, default=LAMBDA_IMG_MASK, help="Weight of the image masked-feature reconstruction loss")
    parser.add_argument("--lambda_clim_mask", type=float, default=LAMBDA_CLIM_MASK, help="Weight of the climate masked-token reconstruction loss")
    parser.add_argument("--img_mask_ratio", type=float, default=IMG_MASK_RATIO, help="Fraction of image patches to mask during hybrid SSL")
    parser.add_argument("--clim_mask_ratio", type=float, default=CLIM_MASK_RATIO, help="Fraction of climate timesteps to mask during hybrid SSL")
    parser.add_argument("--clim_mask_mode", type=str, default=CLIM_MASK_MODE, choices=["block", "random"], help="Climate masking strategy")
    parser.add_argument("--clim_block_size", type=int, default=CLIM_BLOCK_SIZE, help="Contiguous climate mask span when clim_mask_mode=block")
    parser.add_argument("--img_mask_patch_size", type=int, default=IMG_MASK_PATCH_SIZE, help="Patch size used for input-space image masking")
    parser.add_argument("--img_decoder_hidden", type=int, default=IMG_DECODER_HIDDEN, help="Hidden size of the lightweight image reconstruction head")
    parser.add_argument("--clim_decoder_hidden", type=int, default=CLIM_DECODER_HIDDEN, help="Hidden size of the lightweight climate reconstruction head")
    parser.add_argument("--aux_dropout", type=float, default=AUX_DROPOUT, help="Dropout inside the hybrid SSL auxiliary decoders")

    # Optional explicit path overrides.
    # These let the same train_ssl.py work for both LUCAS and RaCA without editing config.py each time.
    parser.add_argument("--train_l8_dir", type=str, default=None, help="Override path to training raster patches")
    parser.add_argument("--test_l8_dir", type=str, default=None, help="Override path to test raster patches")
    parser.add_argument("--val_l8_dir", type=str, default=None, help="Override path to validation raster patches")
    parser.add_argument("--main_csv_path", type=str, default=None, help="Override main dataset CSV path")
    parser.add_argument("--climate_csv_dir", type=str, default=None, help="Override climate CSV directory path")
    parser.add_argument("--oc_max_override", type=float, default=None, help="Override OC_MAX used only by normalization")
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(train_ds, val_ds, test_ds, train_batch_size, test_batch_size, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dl, test_dl, val_dl


def resolve_paths(args) -> Dict[str, str]:

    resolved = {
        "train_l8_dir": args.train_l8_dir or train_l8_folder_path,
        "test_l8_dir": args.test_l8_dir or test_l8_folder_path,
        "val_l8_dir": args.val_l8_dir or val_l8_folder_path,
        "main_csv_path": args.main_csv_path or lucas_csv_path,
        "climate_csv_dir": args.climate_csv_dir or climate_csv_folder_path,
    }

    missing = [k for k, v in resolved.items() if v is None or str(v).strip() == ""]
    if missing:
        raise ValueError(
            f"Missing required dataset paths: {missing}. "
            "Provide them either in config.py or via --train_l8_dir/--test_l8_dir/--val_l8_dir/--main_csv_path/--climate_csv_dir."
        )

    return resolved


def print_runtime_configuration(args, resolved_paths: Dict[str, str], oc_max: float):
    print(tc.BOLD_BAKGROUNDs.BLUE, "RUNTIME DATA CONFIGURATION", tc.ENDC)
    print(f"dataset            : {args.dataset}")
    print(f"ssl_objective      : {args.ssl_objective}")
    print(f"train_l8_dir       : {resolved_paths['train_l8_dir']}")
    print(f"test_l8_dir        : {resolved_paths['test_l8_dir']}")
    print(f"val_l8_dir         : {resolved_paths['val_l8_dir']}")
    print(f"main_csv_path      : {resolved_paths['main_csv_path']}")
    print(f"climate_csv_dir    : {resolved_paths['climate_csv_dir']}")
    print(f"oc_max(norm only)  : {oc_max}")
    print("")


def main():
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Date and Time:", start_string)

    parser = parse_arguments()
    args = parser.parse_args()

    exp_name = args.experiment_name
    num_workers = args.num_workers
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    lr_scheduler = args.lr_scheduler
    dataset = args.dataset
    use_srtm = args.use_srtm
    use_spatial_attention = args.use_spatial_attention
    cnn_architecture = args.cnn_architecture
    rnn_architecture = args.rnn_architecture
    reg_version = args.reg_version
    use_lstm_branch = args.use_lstm_branch
    seeds = args.seeds

    if not use_lstm_branch:
        raise ValueError(
            "SSL-SoilNet pretraining requires the climate branch. Please add -lstm to enable image-climate SSL."
        )

    resolved_paths = resolve_paths(args)
    oc_max = float(args.oc_max_override) if args.oc_max_override is not None else DEFAULT_OC_MAX[dataset]

    if dataset == "LUCAS":
        from dataset.dataset_loader import SNDatasetClimate, myNormalize, myToTensor, Augmentations
    elif dataset == "RaCA":
        from dataset.dataset_loader_us import SNDatasetClimate, myNormalize, myToTensor, Augmentations
    else:
        raise ValueError("Invalid dataset name")

    print_runtime_configuration(args, resolved_paths, oc_max)

    if use_srtm:
        mynorm = myNormalize(
            img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)], [(12), (-4, 2963)], [(13), (0, 90)]],
            oc_min=0,
            oc_max=oc_max,
        )
    else:
        mynorm = myNormalize(
            img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)]],
            oc_min=0,
            oc_max=oc_max,
        )

    my_to_tensor = myToTensor()
    my_augmentation = Augmentations()
    train_transform = transforms.Compose([mynorm, my_to_tensor, my_augmentation])
    test_transform = transforms.Compose([mynorm, my_to_tensor])

    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if not use_srtm else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    train_ds = SNDatasetClimate(
        resolved_paths["train_l8_dir"],
        resolved_paths["main_csv_path"],
        resolved_paths["climate_csv_dir"],
        l8_bands=bands,
        transform=train_transform,
    )
    test_ds = SNDatasetClimate(
        resolved_paths["test_l8_dir"],
        resolved_paths["main_csv_path"],
        resolved_paths["climate_csv_dir"],
        l8_bands=bands,
        transform=test_transform,
    )
    val_ds = SNDatasetClimate(
        resolved_paths["val_l8_dir"],
        resolved_paths["main_csv_path"],
        resolved_paths["climate_csv_dir"],
        l8_bands=bands,
        transform=test_transform,
    )
    test_ds_w_id = SNDatasetClimate(
        resolved_paths["test_l8_dir"],
        resolved_paths["main_csv_path"],
        resolved_paths["climate_csv_dir"],
        l8_bands=bands,
        transform=test_transform,
        return_point_id=True,
    )

    seq_len = test_ds_w_id[0][0][1].shape[0]
    csv_files = [f for f in os.listdir(resolved_paths["climate_csv_dir"]) if f.endswith(".csv")]
    num_climate_features = len(csv_files)

    if args.ssl_objective == "hybrid":
        loss_fn = HybridMultiModalSSLoss(
            temperature=args.temperature,
            lambda_img_mask=args.lambda_img_mask,
            lambda_clim_mask=args.lambda_clim_mask,
        )
    else:
        loss_fn = SimCLR(temperature=args.temperature)

    cv_results = {
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

    run_name = datetime.now().strftime("D_%Y_%m_%d_T_%H_%M")
    print("Run Name:", run_name)

    best_seed = None
    best_val_loss = float("inf")
    best_model_path = None
    seed_summaries = []

    for idx, seed in enumerate(seeds):
        print(tc.BOLD_BAKGROUNDs.PURPLE, f"CROSS VAL {idx + 1} | seed={seed}", tc.ENDC)
        set_seed(seed)

        train_dl, test_dl, val_dl = build_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            train_batch_size,
            test_batch_size,
            num_workers,
            seed,
        )

        model = SoilNetSimCLR(
            use_glam=use_spatial_attention,
            cnn_arch=cnn_architecture,
            reg_version=reg_version,
            cnn_in_channels=len(bands),
            regresor_input_from_cnn=128,
            lstm_n_features=num_climate_features,
            lstm_n_layers=2,
            lstm_out=128,
            hidden_size=128,
            rnn_arch=rnn_architecture,
            seq_len=seq_len,
            use_hybrid_ssl=(args.ssl_objective == "hybrid"),
            img_mask_ratio=args.img_mask_ratio,
            clim_mask_ratio=args.clim_mask_ratio,
            clim_mask_mode=args.clim_mask_mode,
            clim_block_size=args.clim_block_size,
            img_mask_patch_size=args.img_mask_patch_size,
            img_decoder_hidden=args.img_decoder_hidden,
            clim_decoder_hidden=args.clim_decoder_hidden,
            aux_dropout=args.aux_dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        results = train(
            model,
            train_dl,
            test_dl,
            val_dl,
            optimizer,
            loss_fn,
            epochs=num_epochs,
            lr_scheduler=lr_scheduler,
        )

        cv_results["train_loss"].append(results["train_loss"])
        cv_results["train_acc_top1"].append(results["train_acc_top1"])
        cv_results["train_acc_top5"].append(results["train_acc_top5"])
        cv_results["train_acc_mean_pos"].append(results["train_acc_mean_pos"])
        cv_results["val_loss"].append(results["val_loss"])
        cv_results["val_acc_top1"].append(results["val_acc_top1"])
        cv_results["val_acc_top5"].append(results["val_acc_top5"])
        cv_results["val_acc_mean_pos"].append(results["val_acc_mean_pos"])
        cv_results["train_loss_components"].append(results["train_loss_components"])
        cv_results["val_loss_components"].append(results["val_loss_components"])

        seed_model_path = f"results/RUN_{exp_name}_{run_name}_seed{seed}_SelfSupervised.pth"
        torch.save(model, seed_model_path)

        seed_best_val = float(min(results["val_loss"])) if results["val_loss"] else float("inf")
        seed_best_epoch = int(np.argmin(np.asarray(results["val_loss"]))) + 1 if results["val_loss"] else None
        seed_summary: Dict = {
            "seed": int(seed),
            "best_val_loss": seed_best_val,
            "best_epoch": seed_best_epoch,
            "model_path": seed_model_path,
        }
        if results["val_loss_components"]:
            seed_summary["best_epoch_val_loss_components"] = results["val_loss_components"][seed_best_epoch - 1]
        seed_summaries.append(seed_summary)

        if seed_best_val < best_val_loss:
            best_val_loss = seed_best_val
            best_seed = seed
            best_model_path = f"results/RUN_{exp_name}_{run_name}_SelfSupervised_best.pth"
            torch.save(model, best_model_path)

    # Keep a stable default path as well, pointing to the best seed model.
    final_default_model_path = f"results/RUN_{exp_name}_{run_name}_SelfSupervised.pth"
    if best_model_path is not None:
        shutil.copyfile(best_model_path, final_default_model_path)

    train_arr = np.asarray(cv_results["train_loss"])
    val_arr = np.asarray(cv_results["val_loss"])
    plot_train_test_losses(
        train_arr,
        val_arr,
        title="SSL Total Loss",
        x_label="Epochs",
        y_label="Loss",
        min_max_bounds=True,
        tight_x_lim=True,
        train_legend="Train",
        test_legend="Validation",
        save_path=f"results/RUN_{exp_name}_{run_name}_ssl_total_loss.png",
        show=False,
    )

    train_arr = np.asarray(cv_results["train_acc_mean_pos"])
    val_arr = np.asarray(cv_results["val_acc_mean_pos"])
    plot_train_test_losses(
        train_arr,
        val_arr,
        title="Average Self-Rank",
        x_label="Epochs",
        y_label="Rank",
        min_max_bounds=True,
        tight_x_lim=True,
        train_legend="Train",
        test_legend="Validation",
        save_path=f"results/RUN_{exp_name}_{run_name}_ssl_rank.png",
        show=False,
    )

    train_arr = np.asarray(cv_results["train_acc_top5"])
    val_arr = np.asarray(cv_results["val_acc_top5"])
    plot_train_test_losses(
        train_arr,
        val_arr,
        title="Top 5 Probability",
        x_label="Epochs",
        y_label="Top 5 Probability",
        min_max_bounds=True,
        tight_x_lim=True,
        train_legend="Train",
        test_legend="Validation",
        save_path=f"results/RUN_{exp_name}_{run_name}_ssl_top5.png",
        show=False,
    )

    train_arr = np.asarray(cv_results["train_acc_top1"])
    val_arr = np.asarray(cv_results["val_acc_top1"])
    plot_train_test_losses(
        train_arr,
        val_arr,
        title="Top 1 Probability",
        x_label="Epochs",
        y_label="Top 1 Probability",
        min_max_bounds=True,
        tight_x_lim=True,
        train_legend="Train",
        test_legend="Validation",
        save_path=f"results/RUN_{exp_name}_{run_name}_ssl_top1.png",
        show=False,
    )

    summary = {
        "run_name": run_name,
        "dataset": dataset,
        "ssl_objective": args.ssl_objective,
        "best_seed": int(best_seed) if best_seed is not None else None,
        "best_val_loss": float(best_val_loss) if best_seed is not None else None,
        "best_model_path": best_model_path,
        "default_model_path": final_default_model_path,
        "resolved_paths": resolved_paths,
        "seeds": [int(s) for s in seeds],
        "seed_summaries": seed_summaries,
        "args": vars(args),
        "cv_results": cv_results,
    }
    with open(f"results/RUN_{exp_name}_{run_name}_ssl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(tc.BOLD_BAKGROUNDs.GREEN, "TRAINING FINISHED", tc.ENDC)
    if best_seed is not None:
        print(tc.OKBLUE, f"Best seed: {best_seed} | Best val loss: {best_val_loss:.6f}", tc.ENDC)
        print(tc.OKBLUE, f"Best SSL model saved to: {best_model_path}", tc.ENDC)
        print(tc.OKBLUE, f"Default SSL model saved to: {final_default_model_path}", tc.ENDC)


if __name__ == "__main__":
    main()
