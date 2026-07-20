import argparse
import copy
import json
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from soilnet.soil_net import SoilNet, SoilNetJustLSTM, SoilNetLSTM, SoilNetSimCLRwRegHead
from train_utils import (
    RMSELoss,
    RMSLELoss,
    evaluate_regression_metrics,
    load_checkpoint,
    test_step_w_id,
    train,
)


if not os.path.exists('results'):
    os.mkdir('results')

now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")

device = "cuda" if torch.cuda.is_available() else "cpu"

train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path
SIMCLR_PATH = config.SIMCLR_PATH

EXP_NAME = 'LUCAS_Transformer_NoImage'
DATASET = 'LUCAS'
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
LR_SCHEDULER = "step"
USE_SRTM = False
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = "ViT"
RNN_ARCHITECTURE = 'Transformer'
REG_VERSION = 1
SEEDS = [1]
USE_LSTM_BRANCH = False
LOG_LOSS = False
SAVE_TRAIN_DATA_METRICS = False
LOAD_SIMCLR_MODEL = False
JUST_LSTM = False
FUSION_MODE = 'late'
GATE_HIDDEN_SIZE = 64
GATE_DROPOUT = 0.10

ROBUST_MODE = 'none'
ROBUST_MODALITY = 'climate'
ROBUST_DROP_PROB = 0.12
ROBUST_IMG_SEVERITY = 0.30
ROBUST_CLIM_SEVERITY = 0.40
ROBUST_SUPERVISED_WEIGHT = 0.12
ROBUST_CONSISTENCY_WEIGHT = 0.05
ROBUST_WARMUP_EPOCHS = 0
ROBUST_BATCH_PROB = 0.35
ROBUST_MIN_SAMPLES = 4
ROBUST_GATE_WEIGHT = 0.08
ROBUST_GATE_MARGIN = 0.10
ROBUST_TEACHER_BLEND = 0.25
ROBUST_PROTOTYPE_MOMENTUM = 0.90
ROBUST_STOPGRAD_AUX = True
ROBUST_FREEZE_AUX_STOCHASTIC = True

ROBUST_MODE_CLI_FLAGS = {'-rm', '--robust_mode'}
ROBUST_HPARAM_CLI_FLAGS = {
    '-rmod', '--robust_modality',
    '-rdp', '--robust_drop_prob',
    '-rir', '-ris', '--robust_img_mask_ratio', '--robust_img_severity',
    '-rcr', '-rcs', '--robust_clim_mask_ratio', '--robust_clim_severity',
    '-rsw', '--robust_supervised_weight',
    '-rcw', '--robust_consistency_weight',
    '-rwe', '--robust_warmup_epochs',
    '-rbp', '--robust_batch_prob',
    '-rms', '--robust_min_samples',
    '-rgw', '--robust_gate_weight',
    '-rgm', '--robust_gate_margin',
    '-rtb', '--robust_teacher_blend',
    '-rpm', '--robust_prototype_momentum',
    '--robust_no_stopgrad',
    '--robust_disable_aux_freeze',
}


class RobustMode:
    NONE = 'none'
    FEATURE_FALLBACK = 'feature_fallback'
    LEGACY_ALIAS = 'modality_dropout'


def parse_arguments(raw_argv=None):
    parser = argparse.ArgumentParser(description='SoilNet Training')
    parser.add_argument('-e', '--exp_name', type=str, default=EXP_NAME, help='Experiment name - helps to identify the experiment')
    parser.add_argument('-d', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA'], help='Dataset name to use')
    parser.add_argument('-w', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
    parser.add_argument('-trbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('-tsbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
    parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
    parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
    parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT', 'resnet50'], help='CNN architecture')
    parser.add_argument('-rnn', '--rnn_architecture', type=str, default=RNN_ARCHITECTURE, choices=['LSTM', 'GRU', 'RNN', 'Transformer'], help='RNN architecture')
    parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, help='Regression version')
    parser.add_argument('-seed', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross-validation. input example: 1 2 3 4 5')
    parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Climate data - I know! the name is misleading')
    parser.add_argument('-log', '--log_loss', action='store_true', default=LOG_LOSS, help='Use logarithmic loss')
    parser.add_argument('-stm', '--save_train_data_metrics', action='store_true', default=SAVE_TRAIN_DATA_METRICS, help='Save training data metrics')
    parser.add_argument('-simclr', '--load_simclr_model', action='store_true', default=LOAD_SIMCLR_MODEL, help='Load Self-supervised model to fine-tune')
    parser.add_argument('-jlstm', '--just_lstm', action='store_true', default=JUST_LSTM, help='Use only climate data')
    parser.add_argument('-fm', '--fusion_mode', type=str, default=FUSION_MODE, choices=['late', 'gated'], help='Fusion head for SSL fine-tuning. late = original late fusion; gated = dynamic sample-wise gating head.')
    parser.add_argument('-gh', '--gate_hidden_size', type=int, default=GATE_HIDDEN_SIZE, help='Hidden width of the dynamic gate MLP (used only when --fusion_mode gated and --load_simclr_model are enabled).')
    parser.add_argument('-gd', '--gate_dropout', type=float, default=GATE_DROPOUT, help='Dropout for the dynamic fusion head (used only when --fusion_mode gated and --load_simclr_model are enabled).')

    parser.add_argument(
        '-rm', '--robust_mode',
        type=str,
        default=ROBUST_MODE,
        choices=[RobustMode.NONE, RobustMode.FEATURE_FALLBACK, RobustMode.LEGACY_ALIAS],
        help='Third innovation master switch. none = original fine-tuning; feature_fallback = feature-space degraded-modality robust multimodal learning. legacy alias modality_dropout is still accepted and mapped to feature_fallback.',
    )
    parser.add_argument('-rmod', '--robust_modality', type=str, default=ROBUST_MODALITY, choices=['random_one', 'image', 'climate', 'both'], help='Which modality to degrade during robust multimodal fine-tuning.')
    parser.add_argument('-rdp', '--robust_drop_prob', type=float, default=ROBUST_DROP_PROB, help='Probability that a training sample is selected for the robust branch inside an activated batch.')
    parser.add_argument('-rir', '-ris', '--robust_img_mask_ratio', '--robust_img_severity', dest='robust_img_severity', type=float, default=ROBUST_IMG_SEVERITY, help='Image degradation severity in feature space. 0 = keep image embedding unchanged; 1 = fully replace by the running image prototype. Old img-mask flag names are kept as aliases for backward compatibility.')
    parser.add_argument('-rcr', '-rcs', '--robust_clim_mask_ratio', '--robust_clim_severity', dest='robust_clim_severity', type=float, default=ROBUST_CLIM_SEVERITY, help='Climate degradation severity in feature space. 0 = keep climate embedding unchanged; 1 = fully replace by the running climate prototype. Old climate-mask flag names are kept as aliases for backward compatibility.')
    parser.add_argument('-rsw', '--robust_supervised_weight', type=float, default=ROBUST_SUPERVISED_WEIGHT, help='Weight of the degraded-branch soft teacher loss.')
    parser.add_argument('-rcw', '--robust_consistency_weight', type=float, default=ROBUST_CONSISTENCY_WEIGHT, help='Weight of the degraded/full prediction consistency loss.')
    parser.add_argument('-rwe', '--robust_warmup_epochs', type=int, default=ROBUST_WARMUP_EPOCHS, help='Number of initial epochs that keep the original full-modality objective before enabling the robust branch.')
    parser.add_argument('-rbp', '--robust_batch_prob', type=float, default=ROBUST_BATCH_PROB, help='Probability that a training batch activates the sparse robust branch.')
    parser.add_argument('-rms', '--robust_min_samples', type=int, default=ROBUST_MIN_SAMPLES, help='Minimum number of affected samples required before the robust branch is executed on an activated batch.')
    parser.add_argument('-rgw', '--robust_gate_weight', type=float, default=ROBUST_GATE_WEIGHT, help='Weight of the soft gate-margin regularizer used when fusion_mode=gated.')
    parser.add_argument('-rgm', '--robust_gate_margin', type=float, default=ROBUST_GATE_MARGIN, help='Margin used by the gate regularizer. Larger values make the gate prefer the intact modality more strongly on degraded samples.')
    parser.add_argument('-rtb', '--robust_teacher_blend', type=float, default=ROBUST_TEACHER_BLEND, help='Blend factor between ground truth and full-modality teacher prediction for the degraded auxiliary target. 0 = pure teacher target; 1 = pure ground truth.')
    parser.add_argument('-rpm', '--robust_prototype_momentum', type=float, default=ROBUST_PROTOTYPE_MOMENTUM, help='Momentum used by the running modality prototypes that represent degraded/missing-modality fallback embeddings.')
    parser.add_argument('--robust_no_stopgrad', dest='robust_stopgrad_aux', action='store_false', default=ROBUST_STOPGRAD_AUX, help='Allow the robust branch to update the encoder. By default the robust branch stops gradients into the SSL encoder and only regularizes the fusion head, which is safer when you already have a strong pretrained encoder.')
    parser.add_argument('--robust_disable_aux_freeze', dest='robust_freeze_aux_stochastic', action='store_false', default=ROBUST_FREEZE_AUX_STOCHASTIC, help='Do not freeze dropout/BatchNorm-like stochastic modules during the degraded auxiliary forward. By default they are frozen for the robust branch to reduce noise.')
    return parser.parse_args(raw_argv)


def _detect_present_flags(raw_argv, accepted_flags):
    return [flag for flag in raw_argv if flag in accepted_flags]


def normalize_robust_mode(robust_mode: str) -> str:
    if robust_mode == RobustMode.LEGACY_ALIAS:
        return RobustMode.FEATURE_FALLBACK
    return robust_mode


def resolve_robust_cli_behavior(args, raw_argv):
    explicit_mode_flags = _detect_present_flags(raw_argv, ROBUST_MODE_CLI_FLAGS)
    explicit_hparam_flags = _detect_present_flags(raw_argv, ROBUST_HPARAM_CLI_FLAGS)

    args.robust_mode = normalize_robust_mode(args.robust_mode)
    robust_auto_enabled = False
    if not explicit_mode_flags and args.robust_mode == RobustMode.NONE and len(explicit_hparam_flags) > 0:
        args.robust_mode = RobustMode.FEATURE_FALLBACK
        robust_auto_enabled = True

    return {
        'robust_mode': args.robust_mode,
        'robust_auto_enabled': robust_auto_enabled,
        'explicit_mode_flags': explicit_mode_flags,
        'explicit_hparam_flags': explicit_hparam_flags,
    }


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


def build_model(
    use_spatial_attention: bool,
    cnn_architecture: str,
    reg_version: int,
    bands,
    use_lstm_branch: bool,
    just_lstm: bool,
    num_climate_features: int,
    rnn_architecture: str,
    seq_len: Optional[int],
    load_simclr_model: bool,
    simclr_path: str,
    fusion_mode: str,
    gate_hidden_size: int,
    gate_dropout: float,
):
    if not just_lstm:
        if use_lstm_branch:
            model = SoilNetLSTM(
                use_glam=use_spatial_attention,
                cnn_arch=cnn_architecture,
                reg_version=reg_version,
                cnn_in_channels=len(bands),
                regresor_input_from_cnn=1024,
                lstm_n_features=num_climate_features,
                lstm_n_layers=2,
                lstm_out=128,
                hidden_size=128,
                rnn_arch=rnn_architecture,
                seq_len=seq_len,
            ).to(device)
        else:
            model = SoilNet(
                use_glam=use_spatial_attention,
                cnn_arch=cnn_architecture,
                reg_version=reg_version,
                cnn_in_channels=len(bands),
                regresor_input_from_cnn=1024,
                hidden_size=128,
            ).to(device)
    else:
        model = SoilNetJustLSTM(
            use_glam=use_spatial_attention,
            cnn_arch=cnn_architecture,
            reg_version=reg_version,
            cnn_in_channels=len(bands),
            regresor_input_from_cnn=1024,
            lstm_n_features=num_climate_features,
            lstm_n_layers=2,
            lstm_out=128,
            hidden_size=128,
            rnn_arch=rnn_architecture,
            seq_len=seq_len,
        ).to(device)

    if load_simclr_model:
        ssl_model = torch.load(simclr_path, map_location=device, weights_only=False)
        ssl_model = ssl_model.to(device)
        model = SoilNetSimCLRwRegHead(
            ssl_model,
            hidden_size=128,
            reg_version=reg_version,
            fusion_mode=fusion_mode,
            gate_hidden_size=gate_hidden_size,
            gate_dropout=gate_dropout,
        ).to(device)

    return model


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
    val_dl = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl


def is_better_seed(candidate_metrics: Dict, best_metrics: Optional[Dict]) -> bool:
    if best_metrics is None:
        return True

    if candidate_metrics['RMSE'] < best_metrics['RMSE']:
        return True

    if np.isclose(candidate_metrics['RMSE'], best_metrics['RMSE']) and candidate_metrics['CCC'] > best_metrics['CCC']:
        return True

    return False


def to_python_metrics(metrics: Optional[Dict]):
    if metrics is None:
        return None
    output = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            output[k] = v.item()
        else:
            output[k] = v
    return output


def sanitize_curve_array_for_plot(curves) -> np.ndarray:

    arr = np.asarray(curves, dtype=float).copy()

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    for row_idx in range(arr.shape[0]):
        row = arr[row_idx]
        finite_mask = np.isfinite(row)

        if finite_mask.all():
            continue

        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size == 0:
            row[:] = 0.0
            continue

        invalid_idx = np.flatnonzero(~finite_mask)
        row[invalid_idx] = np.interp(invalid_idx, finite_idx, row[finite_idx])

    return arr


def build_robust_training_config(
    robust_mode: str,
    robust_modality: str,
    robust_drop_prob: float,
    robust_img_severity: float,
    robust_clim_severity: float,
    robust_supervised_weight: float,
    robust_consistency_weight: float,
    robust_warmup_epochs: int,
    robust_batch_prob: float,
    robust_min_samples: int,
    robust_gate_weight: float,
    robust_gate_margin: float,
    robust_teacher_blend: float,
    robust_prototype_momentum: float,
    robust_stopgrad_aux: bool,
    robust_freeze_aux_stochastic: bool,
) -> Optional[Dict]:
    robust_mode = normalize_robust_mode(robust_mode)
    if robust_mode == RobustMode.NONE:
        return None

    return {
        'mode': robust_mode,
        'modality': robust_modality,
        'drop_prob': float(robust_drop_prob),
        'img_severity': float(robust_img_severity),
        'clim_severity': float(robust_clim_severity),
        'supervised_weight': float(robust_supervised_weight),
        'consistency_weight': float(robust_consistency_weight),
        'warmup_epochs': int(robust_warmup_epochs),
        'batch_prob': float(robust_batch_prob),
        'min_samples': int(robust_min_samples),
        'gate_weight': float(robust_gate_weight),
        'gate_margin': float(robust_gate_margin),
        'teacher_blend': float(robust_teacher_blend),
        'prototype_momentum': float(robust_prototype_momentum),
        'stopgrad_aux': bool(robust_stopgrad_aux),
        'freeze_aux_stochastic': bool(robust_freeze_aux_stochastic),
    }


def serialize_robust_training_config(robust_training_config: Optional[Dict]) -> Optional[Dict]:
    if robust_training_config is None:
        return None

    clean = {}
    for k, v in robust_training_config.items():
        if str(k).startswith('_'):
            continue
        if isinstance(v, torch.Tensor):
            continue
        clean[k] = v
    return clean


if __name__ == '__main__':
    raw_argv = sys.argv[1:]
    args = parse_arguments(raw_argv)
    robust_cli_info = resolve_robust_cli_behavior(args, raw_argv)

    EXP_NAME = args.exp_name
    DATASET = args.dataset
    NUM_WORKERS = args.num_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    LR_SCHEDULER = args.lr_scheduler
    USE_SRTM = args.use_srtm
    USE_SPATIAL_ATTENTION = args.use_spatial_attention
    CNN_ARCHITECTURE = args.cnn_architecture
    RNN_ARCHITECTURE = args.rnn_architecture
    REG_VERSION = args.reg_version
    SEEDS = args.seeds
    USE_LSTM_BRANCH = args.use_lstm_branch
    LOG_LOSS = args.log_loss
    SAVE_TRAIN_DATA_METRICS = args.save_train_data_metrics
    LOAD_SIMCLR_MODEL = args.load_simclr_model
    JUST_LSTM = args.just_lstm
    FUSION_MODE = args.fusion_mode
    GATE_HIDDEN_SIZE = args.gate_hidden_size
    GATE_DROPOUT = args.gate_dropout

    if DATASET == 'LUCAS':
        from dataset.dataset_loader import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 87
        UNIT = 'g/kg'
    elif DATASET == 'RaCA':
        from dataset.dataset_loader_us import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 4115
        UNIT = 'Mg/ha'
    else:
        raise ValueError('Invalid dataset Name')

    if JUST_LSTM:
        USE_LSTM_BRANCH = True
        USE_SPATIAL_ATTENTION = False

    if LOAD_SIMCLR_MODEL:
        if USE_LSTM_BRANCH is False:
            raise Exception('LOAD_SIMCLR_MODEL is enabled but LSTM branch is disabled. Please enable LSTM branch.')
        if JUST_LSTM:
            raise Exception('LOAD_SIMCLR_MODEL is enabled but JUST_LSTM is enabled. Please disable JUST_LSTM.')

    if LOAD_SIMCLR_MODEL:
        print('\033[91m\033[1m\033[5mWARNING!\033[0m')
        print(
            '\033[93m Loading SimCLR Model is enabled.\n'
            ' This will overwrite chosen architectures.\n'
            ' Also, make sure that LSTM is enabled. \033[0m'
        )

    if FUSION_MODE == 'gated' and not LOAD_SIMCLR_MODEL:
        print('\033[93mWARNING: fusion_mode=gated is currently applied only to the SSL fine-tuning head. Because --load_simclr_model is disabled, the code will fall back to the original supervised architecture.\033[0m')

    if USE_SRTM:
        mynorm = myNormalize(
            img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)], [12, (-4, 2963)], [13, (0, 90)]],
            oc_min=0,
            oc_max=OC_MAX,
        )
    else:
        mynorm = myNormalize(
            img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)]],
            oc_min=0,
            oc_max=OC_MAX,
        )

    my_to_tensor = myToTensor()
    my_augmentation = Augmentations()
    train_transform = transforms.Compose([mynorm, my_to_tensor, my_augmentation])
    test_transform = transforms.Compose([mynorm, my_to_tensor])

    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if not USE_SRTM else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    if not USE_LSTM_BRANCH:
        train_ds = SNDataset(train_l8_folder_path, lucas_csv_path, l8_bands=bands, transform=train_transform)
        test_ds = SNDataset(test_l8_folder_path, lucas_csv_path, l8_bands=bands, transform=test_transform)
        val_ds = SNDataset(val_l8_folder_path, lucas_csv_path, l8_bands=bands, transform=test_transform)
        test_ds_w_id = SNDataset(test_l8_folder_path, lucas_csv_path, l8_bands=bands, transform=test_transform, return_point_id=True)
        SEQ_LEN = None
        NUM_CLIMATE_FEATURES = 0
        CSV_FILES = []
    else:
        train_ds = SNDatasetClimate(
            train_l8_folder_path,
            lucas_csv_path,
            climate_csv_folder_path,
            l8_bands=bands,
            transform=train_transform,
        )
        test_ds = SNDatasetClimate(
            test_l8_folder_path,
            lucas_csv_path,
            climate_csv_folder_path,
            l8_bands=bands,
            transform=test_transform,
        )
        val_ds = SNDatasetClimate(
            val_l8_folder_path,
            lucas_csv_path,
            climate_csv_folder_path,
            l8_bands=bands,
            transform=test_transform,
        )
        test_ds_w_id = SNDatasetClimate(
            test_l8_folder_path,
            lucas_csv_path,
            climate_csv_folder_path,
            l8_bands=bands,
            transform=test_transform,
            return_point_id=True,
        )
        SEQ_LEN = test_ds_w_id[0][0][1].shape[0]
        CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
        NUM_CLIMATE_FEATURES = len(CSV_FILES)

    robust_training_config_base = build_robust_training_config(
        robust_mode=args.robust_mode,
        robust_modality=args.robust_modality,
        robust_drop_prob=args.robust_drop_prob,
        robust_img_severity=args.robust_img_severity,
        robust_clim_severity=args.robust_clim_severity,
        robust_supervised_weight=args.robust_supervised_weight,
        robust_consistency_weight=args.robust_consistency_weight,
        robust_warmup_epochs=args.robust_warmup_epochs,
        robust_batch_prob=args.robust_batch_prob,
        robust_min_samples=args.robust_min_samples,
        robust_gate_weight=args.robust_gate_weight,
        robust_gate_margin=args.robust_gate_margin,
        robust_teacher_blend=args.robust_teacher_blend,
        robust_prototype_momentum=args.robust_prototype_momentum,
        robust_stopgrad_aux=args.robust_stopgrad_aux,
        robust_freeze_aux_stochastic=args.robust_freeze_aux_stochastic,
    )

    if robust_cli_info['robust_auto_enabled']:
        print(
            tc.WARNING,
            'Robust multimodal learning was automatically enabled because robust hyperparameter flags were detected on the command line.',
            tc.ENDC,
        )

    if robust_training_config_base is not None:
        if not LOAD_SIMCLR_MODEL or not USE_LSTM_BRANCH or JUST_LSTM:
            print(
                tc.WARNING,
                'Robust feature-fallback learning requires SSL fine-tuning with both image and climate branches enabled. '
                'Because the current run is not multimodal SSL fine-tuning, the robust branch will be disabled and training will fall back to the original fine-tuning objective.',
                tc.ENDC,
            )
            robust_training_config_base = None
        else:
            print(
                tc.OKBLUE,
                'Robust multimodal learning is enabled. This v5 design no longer corrupts raw image/climate inputs. '
                'Instead, it performs sparse feature-space degraded-modality fallback learning: the SSL encoder is trained on the original full-modality path, while the robust branch only regularizes the fusion head with soft teacher targets and a softer gate-margin objective.',
                tc.ENDC,
            )
            print(
                f"Robust runtime config -> batch_prob={robust_training_config_base['batch_prob']:.3f}, min_samples={robust_training_config_base['min_samples']}, sample_drop_prob={robust_training_config_base['drop_prob']:.3f}, img_severity={robust_training_config_base['img_severity']:.3f}, clim_severity={robust_training_config_base['clim_severity']:.3f}, teacher_blend={robust_training_config_base['teacher_blend']:.3f}, stopgrad_aux={robust_training_config_base['stopgrad_aux']}"
            )
            print(
                f"Robust CLI flags -> mode_flags={robust_cli_info['explicit_mode_flags']}, hparam_flags={robust_cli_info['explicit_hparam_flags']}"
            )

    cv_results = {
        'seed': [],
        'best_epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_MAE': [],
        'val_RMSE': [],
        'val_R2': [],
        'val_RPIQ': [],
        'val_MEC': [],
        'val_CCC': [],
    }

    now = datetime.now()
    run_name = now.strftime('D_%Y_%m_%d_T_%H_%M')
    print('Current Date and Time:', run_name)

    best_seed = None
    best_seed_model_path = None
    best_seed_val_metrics = None

    for idx, seed in enumerate(SEEDS):
        print(tc.BOLD_BAKGROUNDs.PURPLE, f'CROSS VAL {idx + 1} | seed={seed}', tc.ENDC)

        set_seed(seed)
        train_dl, val_dl, test_dl = build_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            TRAIN_BATCH_SIZE,
            TEST_BATCH_SIZE,
            NUM_WORKERS,
            seed,
        )

        model = build_model(
            use_spatial_attention=USE_SPATIAL_ATTENTION,
            cnn_architecture=CNN_ARCHITECTURE,
            reg_version=REG_VERSION,
            bands=bands,
            use_lstm_branch=USE_LSTM_BRANCH,
            just_lstm=JUST_LSTM,
            num_climate_features=NUM_CLIMATE_FEATURES,
            rnn_architecture=RNN_ARCHITECTURE,
            seq_len=SEQ_LEN,
            load_simclr_model=LOAD_SIMCLR_MODEL,
            simclr_path=SIMCLR_PATH,
            fusion_mode=FUSION_MODE,
            gate_hidden_size=GATE_HIDDEN_SIZE,
            gate_dropout=GATE_DROPOUT,
        )

        loss_instance = RMSLELoss() if LOG_LOSS else RMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        seed_model_path = f'results/RUN_{EXP_NAME}_{run_name}_seed_{seed}_best.pth.tar'
        robust_training_config = copy.deepcopy(robust_training_config_base)

        results = train(
            model,
            train_dl,
            test_dl,
            val_dl,
            optimizer,
            loss_instance,
            epochs=NUM_EPOCHS,
            lr_scheduler=LR_SCHEDULER,
            save_model_path=seed_model_path,
            save_train_data_metrics=SAVE_TRAIN_DATA_METRICS,
            oc_max=OC_MAX,
            robust_training_config=robust_training_config,
        )

        seed_best_metrics = results['best_val_metrics']
        if seed_best_metrics is None:
            raise RuntimeError(f'No best validation checkpoint was recorded for seed {seed}.')

        cv_results['seed'].append(int(seed))
        cv_results['best_epoch'].append(int(results['best_epoch']))
        cv_results['train_loss'].append(results['train_loss'])
        cv_results['val_loss'].append(results['val_loss'])
        cv_results['val_MAE'].append(float(seed_best_metrics['MAE']))
        cv_results['val_RMSE'].append(float(seed_best_metrics['RMSE']))
        cv_results['val_R2'].append(float(seed_best_metrics['R2']))
        cv_results['val_RPIQ'].append(float(seed_best_metrics['RPIQ']))
        cv_results['val_MEC'].append(float(seed_best_metrics['MEC']))
        cv_results['val_CCC'].append(float(seed_best_metrics['CCC']))

        if is_better_seed(seed_best_metrics, best_seed_val_metrics):
            best_seed = seed
            best_seed_model_path = seed_model_path
            best_seed_val_metrics = seed_best_metrics
            print(
                tc.BOLD_BAKGROUNDs.GREEN,
                f"Current best seed updated to {best_seed} | val_RMSE={best_seed_val_metrics['RMSE']:.6f} | val_CCC={best_seed_val_metrics['CCC']:.6f} | val_RPIQ={best_seed_val_metrics['RPIQ']:.6f}",
                tc.ENDC,
            )

        print(
            f"Seed {seed} best validation metrics -> "
            f"RMSE: {seed_best_metrics['RMSE']:.6f}, "
            f"CCC: {seed_best_metrics['CCC']:.6f}, "
            f"MAE: {seed_best_metrics['MAE']:.6f}, "
            f"R2: {seed_best_metrics['R2']:.6f}, "
            f"RPIQ: {seed_best_metrics['RPIQ']:.6f}"
        )

    if best_seed_model_path is None:
        raise RuntimeError('No best seed model path was produced.')

    global_best_model_path = f'results/RUN_{EXP_NAME}_{run_name}_best.pth.tar'
    shutil.copyfile(best_seed_model_path, global_best_model_path)

    train_arr = sanitize_curve_array_for_plot(cv_results['train_loss'])
    val_arr = sanitize_curve_array_for_plot(cv_results['val_loss'])

    y_label = 'RMSLE' if LOG_LOSS else 'RMSE'
    plot_train_test_losses(
        train_arr,
        val_arr,
        title='Train/Validation Losses',
        x_label='Epochs',
        y_label=y_label,
        min_max_bounds=True,
        tight_x_lim=True,
        train_legend='Train',
        test_legend='Validation',
        save_path=f'results/RUN_{EXP_NAME}_{run_name}.png',
    )

    now = datetime.now()
    finish_string = now.strftime('%Y-%m-%d %H:%M:%S')
    print('Current Date and Time:', finish_string)

    cv_results_full = {}
    cv_results_full['PROTOCOL_NOTE'] = 'Best checkpoint is selected only on validation RMSE, with validation CCC as tie-breaker. Test set is evaluated once at the very end on the selected best seed.'
    cv_results_full['VAL_MAE_MEAN'] = float(np.mean(cv_results['val_MAE']))
    cv_results_full['VAL_RMSE_MEAN'] = float(np.mean(cv_results['val_RMSE']))
    cv_results_full['VAL_R2_MEAN'] = float(np.mean(cv_results['val_R2']))
    cv_results_full['VAL_RPIQ_MEAN'] = float(np.mean(cv_results['val_RPIQ']))
    cv_results_full['VAL_MEC_MEAN'] = float(np.mean(cv_results['val_MEC']))
    cv_results_full['VAL_CCC_MEAN'] = float(np.mean(cv_results['val_CCC']))
    cv_results_full['LOAD_SIMCLR_MODEL'] = LOAD_SIMCLR_MODEL
    cv_results_full['JUST_LSTM'] = JUST_LSTM
    cv_results_full['USE_LSTM_BRANCH'] = USE_LSTM_BRANCH
    cv_results_full['LOG_LOSS'] = LOG_LOSS
    cv_results_full['FUSION_MODE'] = FUSION_MODE
    cv_results_full['GATE_HIDDEN_SIZE'] = GATE_HIDDEN_SIZE if LOAD_SIMCLR_MODEL and FUSION_MODE == 'gated' else None
    cv_results_full['GATE_DROPOUT'] = GATE_DROPOUT if LOAD_SIMCLR_MODEL and FUSION_MODE == 'gated' else None
    cv_results_full['NUM_CLIMATE_FEATURES'] = NUM_CLIMATE_FEATURES if USE_LSTM_BRANCH else None
    cv_results_full['CSV_FILES'] = CSV_FILES if USE_LSTM_BRANCH else None
    cv_results_full['NUM_WORKERS'] = NUM_WORKERS
    cv_results_full['TRAIN_BATCH_SIZE'] = TRAIN_BATCH_SIZE
    cv_results_full['TEST_BATCH_SIZE'] = TEST_BATCH_SIZE
    cv_results_full['LEARNING_RATE'] = LEARNING_RATE
    cv_results_full['NUM_EPOCHS'] = NUM_EPOCHS
    cv_results_full['LR_SCHEDULER'] = LR_SCHEDULER
    cv_results_full['CNN_ARCHITECTURE'] = CNN_ARCHITECTURE
    cv_results_full['RNN_ARCHITECTURE'] = RNN_ARCHITECTURE
    cv_results_full['REG_VERSION'] = REG_VERSION
    cv_results_full['USE_SPATIAL_ATTENTION'] = USE_SPATIAL_ATTENTION
    cv_results_full['BEST_SEED'] = int(best_seed)
    cv_results_full['BEST_SEED_MODEL_PATH'] = best_seed_model_path
    cv_results_full['GLOBAL_BEST_MODEL_PATH'] = global_best_model_path
    cv_results_full['SEEDS'] = SEEDS
    cv_results_full['OC_MAX'] = OC_MAX
    cv_results_full['UNIT'] = UNIT
    cv_results_full['USE_SRTM'] = USE_SRTM
    cv_results_full['TIME'] = {'start': start_string, 'finish': finish_string}
    cv_results_full['best_seed_val_metrics'] = to_python_metrics(best_seed_val_metrics)
    cv_results_full['ROBUST_TRAINING_CONFIG'] = serialize_robust_training_config(robust_training_config_base)
    cv_results_full['cv_results'] = cv_results

    best_model = build_model(
        use_spatial_attention=USE_SPATIAL_ATTENTION,
        cnn_architecture=CNN_ARCHITECTURE,
        reg_version=REG_VERSION,
        bands=bands,
        use_lstm_branch=USE_LSTM_BRANCH,
        just_lstm=JUST_LSTM,
        num_climate_features=NUM_CLIMATE_FEATURES,
        rnn_architecture=RNN_ARCHITECTURE,
        seq_len=SEQ_LEN,
        load_simclr_model=LOAD_SIMCLR_MODEL,
        simclr_path=SIMCLR_PATH,
        fusion_mode=FUSION_MODE,
        gate_hidden_size=GATE_HIDDEN_SIZE,
        gate_dropout=GATE_DROPOUT,
    )
    load_checkpoint(model=best_model, optimizer=None, filename=global_best_model_path)
    best_model.eval()
    print('Best validation-selected model loaded')
    print(f'Global best seed selected from validation: {best_seed}')

    test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    _, best_df = test_step_w_id(
        model=best_model,
        data_loader=test_dl_w_id,
        loss_fn=nn.L1Loss(),
        verbose=False,
        csv_file=f'results/RUN_{EXP_NAME}_{run_name}_best.csv',
    )
    print(f'Best model predictions saved to results/RUN_{EXP_NAME}_{run_name}_best.csv')

    y_true = best_df['y_real'].to_numpy(dtype=float) * OC_MAX
    y_pred = best_df['y_pred'].to_numpy(dtype=float) * OC_MAX
    rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

    final_test_metrics = {
        'RMSE': float(rmse),
        'R2': float(r2),
        'RPIQ': float(rpiq),
        'MAE': float(mae),
        'MEC': float(mec),
        'CCC': float(ccc),
        'unit': UNIT,
        'OC_MAX_used': OC_MAX,
    }

    cv_results_full['final_test_metrics_of_best_seed'] = final_test_metrics

    print('Final test metrics of validation-selected best seed ->')
    print(
        f"RMSE: {final_test_metrics['RMSE']:.6f}, "
        f"MAE: {final_test_metrics['MAE']:.6f}, "
        f"R2: {final_test_metrics['R2']:.6f}, "
        f"RPIQ: {final_test_metrics['RPIQ']:.6f}, "
        f"CCC: {final_test_metrics['CCC']:.6f}, "
        f"MEC: {final_test_metrics['MEC']:.6f}, "
        f"unit: {final_test_metrics['unit']}"
    )

    with open(f'results/Metrics_{EXP_NAME}_{run_name}.json', 'w') as fp:
        json.dump(cv_results_full, fp, indent=4)

    with open(f'results/RUN_{EXP_NAME}_{run_name}.json', 'w') as fp:
        json.dump(cv_results_full, fp, indent=4)
