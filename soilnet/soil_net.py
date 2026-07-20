import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from submodules.cnn_feature_extractor import (
    CNNFlattener64,
    CNNFlattener128,
    ResNet101,
    ResNet101GLAM,
    ResNet50,
    VGG16,
    VGG16GLAM,
)
from submodules.vit import VisionTransformer as ViT
from submodules.regressor import Regressor, MultiHeadRegressor
from submodules import rnn
from submodules.src.transformer.transformer import TSTransformerEncoderClassiregressor


class SoilNet(nn.Module):
    def __init__(
        self,
        use_glam: bool = False,
        cnn_arch: str = "resnet101",
        reg_version: int = 1,
        cnn_in_channels: int = 14,
        regresor_input_from_cnn: int = 1024,
        hidden_size: int = 128,
        img_size: int = 64,
    ):
        super().__init__()
        self.cnn = _build_cnn_encoder(
            use_glam=use_glam,
            cnn_arch=cnn_arch,
            cnn_in_channels=cnn_in_channels,
            out_nodes=regresor_input_from_cnn,
            img_size=img_size,
        )
        self.reg = MultiHeadRegressor(
            regresor_input_from_cnn,
            hidden_size=hidden_size,
            version=reg_version,
        )

    def forward(self, raster_stack: torch.Tensor) -> torch.Tensor:
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster)
        return output


class SoilNetLSTM(nn.Module):
    def __init__(
        self,
        use_glam: bool = False,
        cnn_arch: str = "resnet101",
        reg_version: int = 1,
        cnn_in_channels: int = 14,
        regresor_input_from_cnn: int = 1024,
        lstm_n_features: int = 10,
        lstm_n_layers: int = 2,
        lstm_out: int = 128,
        hidden_size: int = 128,
        rnn_arch: str = "LSTM",
        seq_len: int = 61,
        img_size: int = 64,
    ):
        super().__init__()
        self.cnn = _build_cnn_encoder(
            use_glam=use_glam,
            cnn_arch=cnn_arch,
            cnn_in_channels=cnn_in_channels,
            out_nodes=regresor_input_from_cnn,
            img_size=img_size,
        )
        self.lstm = _build_temporal_encoder(
            rnn_arch=rnn_arch,
            lstm_n_features=lstm_n_features,
            hidden_size=hidden_size,
            lstm_n_layers=lstm_n_layers,
            lstm_out=lstm_out,
            seq_len=seq_len,
        )
        self.reg = MultiHeadRegressor(
            regresor_input_from_cnn,
            lstm_out,
            hidden_size=hidden_size,
            version=reg_version,
        )

    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raster_stack, ts_features = input_raster_ts
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        output = self.reg(flat_raster, lstm_output)
        return output


class SoilNetJustLSTM(SoilNetLSTM):
    """This class disables the CNN pathway to use only climate data."""

    def __init__(
        self,
        use_glam: bool = False,
        cnn_arch: str = "resnet101",
        reg_version: int = 1,
        cnn_in_channels: int = 14,
        regresor_input_from_cnn: int = 1024,
        lstm_n_features: int = 10,
        lstm_n_layers: int = 2,
        lstm_out: int = 128,
        hidden_size: int = 128,
        rnn_arch: str = "LSTM",
        seq_len: int = 61,
        img_size: int = 64,
    ):
        super().__init__(
            use_glam=use_glam,
            cnn_arch=cnn_arch,
            reg_version=reg_version,
            cnn_in_channels=cnn_in_channels,
            regresor_input_from_cnn=regresor_input_from_cnn,
            lstm_n_features=lstm_n_features,
            lstm_n_layers=lstm_n_layers,
            lstm_out=lstm_out,
            hidden_size=hidden_size,
            rnn_arch=rnn_arch,
            seq_len=seq_len,
            img_size=img_size,
        )
        self.cnn = None
        self.reg = MultiHeadRegressor(lstm_out, hidden_size=hidden_size, version=reg_version)

    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, ts_features = input_raster_ts
        lstm_output = self.lstm(ts_features)
        output = self.reg(lstm_output)
        return output


class SoilNetSimCLR(nn.Module):
    """
    Self-supervised multimodal encoder used during SSL pretraining.

    Baseline behavior:
        returns image and climate embeddings for cross-modal InfoNCE.

    Hybrid SSL behavior (when return_aux=True and use_hybrid_ssl=True):
        - keeps the original cross-modal contrastive embeddings;
        - masks image patches and reconstructs the original image embedding;
        - masks climate timesteps in contiguous blocks and reconstructs the
          original climate sequence on the masked timesteps.

    This implementation is intentionally lightweight so it fits consumer GPUs:
        - image branch uses feature-level masked reconstruction, not full-pixel decoding;
        - climate branch reconstructs masked sequence tokens from the masked climate embedding.
    """

    def __init__(
        self,
        use_glam: bool = False,
        cnn_arch: str = "resnet101",
        reg_version: int = 1,
        cnn_in_channels: int = 14,
        regresor_input_from_cnn: int = 128,
        lstm_n_features: int = 10,
        lstm_n_layers: int = 2,
        lstm_out: int = 128,
        hidden_size: int = 128,
        rnn_arch: str = "LSTM",
        seq_len: int = 61,
        img_size: int = 64,
        use_hybrid_ssl: bool = False,
        img_mask_ratio: float = 0.35,
        clim_mask_ratio: float = 0.25,
        clim_mask_mode: str = "block",
        clim_block_size: int = 3,
        img_mask_patch_size: int = 8,
        img_decoder_hidden: int = 256,
        clim_decoder_hidden: int = 256,
        aux_dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = _build_cnn_encoder(
            use_glam=use_glam,
            cnn_arch=cnn_arch,
            cnn_in_channels=cnn_in_channels,
            out_nodes=regresor_input_from_cnn,
            img_size=img_size,
        )
        self.lstm = _build_temporal_encoder(
            rnn_arch=rnn_arch,
            lstm_n_features=lstm_n_features,
            hidden_size=hidden_size,
            lstm_n_layers=lstm_n_layers,
            lstm_out=lstm_out,
            seq_len=seq_len,
        )

        self.use_hybrid_ssl = use_hybrid_ssl
        self.img_mask_ratio = float(img_mask_ratio)
        self.clim_mask_ratio = float(clim_mask_ratio)
        self.clim_mask_mode = clim_mask_mode
        self.clim_block_size = int(clim_block_size)
        self.img_mask_patch_size = int(img_mask_patch_size)
        self.img_size = int(img_size)
        self.seq_len = int(seq_len)
        self.lstm_n_features = int(lstm_n_features)
        self.img_embedding_dim = int(regresor_input_from_cnn)
        self.clim_embedding_dim = int(lstm_out)

        # Lightweight reconstruction heads used only during hybrid SSL pretraining.
        self.img_feature_decoder = nn.Sequential(
            nn.LayerNorm(self.img_embedding_dim),
            nn.Linear(self.img_embedding_dim, img_decoder_hidden),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(img_decoder_hidden, self.img_embedding_dim),
        )

        self.clim_token_decoder = nn.Sequential(
            nn.LayerNorm(self.clim_embedding_dim),
            nn.Linear(self.clim_embedding_dim, clim_decoder_hidden),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(clim_decoder_hidden, self.seq_len * self.lstm_n_features),
        )

    def forward(
        self,
        input_raster_ts: Tuple[torch.Tensor, torch.Tensor],
        return_aux: bool = False,
    ):
        raster_stack, ts_features = input_raster_ts

        img_embed = self.cnn(raster_stack)
        clim_embed = self.lstm(ts_features)

        if not return_aux or not self.use_hybrid_ssl:
            return img_embed, clim_embed

        masked_raster, img_patch_mask = self._apply_image_mask(raster_stack)
        masked_ts, clim_time_mask = self._apply_climate_mask(ts_features)

        masked_img_embed = self.cnn(masked_raster)
        masked_clim_embed = self.lstm(masked_ts)

        img_recon = self.img_feature_decoder(masked_img_embed)
        clim_recon = self.clim_token_decoder(masked_clim_embed).view(
            ts_features.shape[0], self.seq_len, self.lstm_n_features
        )

        return {
            "z_img": img_embed,
            "z_clim": clim_embed,
            "img_recon": img_recon,
            "img_target": img_embed.detach(),
            "img_patch_mask": img_patch_mask,
            "clim_recon": clim_recon,
            "clim_target": ts_features.detach(),
            "clim_time_mask": clim_time_mask,
        }

    def _apply_image_mask(self, raster_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.img_mask_ratio <= 0:
            batch_size = raster_stack.shape[0]
            n_patches = (self.img_size // self.img_mask_patch_size) ** 2
            empty_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=raster_stack.device)
            return raster_stack, empty_mask

        batch_size, _, height, width = raster_stack.shape
        patch = self.img_mask_patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by img_mask_patch_size={patch}."
            )

        grid_h = height // patch
        grid_w = width // patch
        n_patches = grid_h * grid_w
        n_mask = max(1, int(round(n_patches * self.img_mask_ratio)))

        patch_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=raster_stack.device)
        for b in range(batch_size):
            idx = torch.randperm(n_patches, device=raster_stack.device)[:n_mask]
            patch_mask[b, idx] = True

        spatial_mask = patch_mask.view(batch_size, grid_h, grid_w)
        spatial_mask = spatial_mask.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)
        spatial_mask = spatial_mask.unsqueeze(1)

        masked_raster = raster_stack.clone()
        masked_raster = masked_raster.masked_fill(spatial_mask, 0.0)
        return masked_raster, patch_mask

    def _apply_climate_mask(self, ts_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.clim_mask_ratio <= 0:
            empty_mask = torch.zeros(ts_features.shape[0], ts_features.shape[1], dtype=torch.bool, device=ts_features.device)
            return ts_features, empty_mask

        batch_size, seq_len, _ = ts_features.shape
        n_mask_steps = max(1, int(round(seq_len * self.clim_mask_ratio)))
        time_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=ts_features.device)

        for b in range(batch_size):
            if self.clim_mask_mode == "block":
                filled = 0
                while filled < n_mask_steps:
                    span = min(self.clim_block_size, n_mask_steps - filled)
                    max_start = max(seq_len - span, 0)
                    start = int(torch.randint(0, max_start + 1, (1,), device=ts_features.device).item())
                    time_mask[b, start:start + span] = True
                    filled = int(time_mask[b].sum().item())
            elif self.clim_mask_mode == "random":
                idx = torch.randperm(seq_len, device=ts_features.device)[:n_mask_steps]
                time_mask[b, idx] = True
            else:
                raise ValueError("clim_mask_mode must be one of ['block', 'random']")

        masked_ts = ts_features.clone()
        masked_ts[time_mask.unsqueeze(-1).expand_as(masked_ts)] = 0.0
        return masked_ts, time_mask


class DynamicGatedFusionRegressor(nn.Module):
    """
    Lightweight sample-wise dynamic fusion head.

    Design goals:
    1) keep the pretrained image/climate encoders untouched;
    2) let each sample adaptively weigh image vs. climate embeddings;
    3) initialize conservatively so training starts close to equal-weight late fusion.
    """

    def __init__(
        self,
        img_dim: int,
        clim_dim: int,
        hidden_size: int = 128,
        gate_hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_dim = img_dim
        self.clim_dim = clim_dim
        self.last_gate_weights: Optional[torch.Tensor] = None

        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + clim_dim, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 2),
        )

        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        fusion_dim = 2 * (img_dim + clim_dim)
        mid_dim = max(hidden_size // 2, 32)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, img_feat: torch.Tensor, clim_feat: torch.Tensor) -> torch.Tensor:
        if img_feat.ndim != 2 or clim_feat.ndim != 2:
            raise ValueError(
                f"DynamicGatedFusionRegressor expects 2D embeddings, got img={img_feat.shape}, clim={clim_feat.shape}."
            )
        if img_feat.shape[0] != clim_feat.shape[0]:
            raise ValueError(
                f"Batch size mismatch between modalities: img={img_feat.shape}, clim={clim_feat.shape}."
            )

        gate_input = torch.cat([img_feat, clim_feat], dim=1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=1)
        img_gate = gate_weights[:, 0:1]
        clim_gate = gate_weights[:, 1:2]

        raw_fused = torch.cat([img_feat, clim_feat], dim=1)
        weighted_fused = torch.cat([img_feat * img_gate, clim_feat * clim_gate], dim=1)
        fused = torch.cat([raw_fused, weighted_fused], dim=1)

        self.last_gate_weights = gate_weights.detach()
        return self.reg_head(fused)

    def get_last_gate_weights(self) -> Optional[torch.Tensor]:
        return self.last_gate_weights


class SoilNetSimCLRwRegHead(nn.Module):
    def __init__(
        self,
        soilnet_simclr: SoilNetSimCLR,
        hidden_size: int = 128,
        reg_version: int = 1,
        fusion_mode: str = "late",
        gate_hidden_size: int = 64,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.soilnet_simclr = soilnet_simclr
        self.fusion_mode = fusion_mode

        if fusion_mode == "late":
            self.reg = MultiHeadRegressor(hidden_size, hidden_size, hidden_size=hidden_size, version=reg_version)
        elif fusion_mode == "gated":
            self.reg = DynamicGatedFusionRegressor(
                img_dim=hidden_size,
                clim_dim=hidden_size,
                hidden_size=hidden_size,
                gate_hidden_size=gate_hidden_size,
                dropout=gate_dropout,
            )
        else:
            raise ValueError("fusion_mode must be one of ['late', 'gated']")

    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raster_stack, ts_features = input_raster_ts
        flat_raster, lstm_output = self.soilnet_simclr((raster_stack, ts_features), return_aux=False)
        output = self.reg(flat_raster, lstm_output)
        return output

    def get_last_gate_weights(self) -> Optional[torch.Tensor]:
        if hasattr(self.reg, "get_last_gate_weights"):
            return self.reg.get_last_gate_weights()
        return None


def _build_cnn_encoder(
    use_glam: bool,
    cnn_arch: str,
    cnn_in_channels: int,
    out_nodes: int,
    img_size: int,
):
    if use_glam:
        if cnn_arch == "resnet101":
            return ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=out_nodes)
        if cnn_arch == "vgg16":
            return VGG16(in_channels=cnn_in_channels, out_nodes=out_nodes)
        if cnn_arch == "ViT":
            raise ValueError("ViT is not supported when GLAM is enabled. Please disable GLAM for ViT.")
        raise ValueError("Invalid CNN architecture. Please choose from 'resnet101' or 'vgg16'.")

    if cnn_arch == "resnet101":
        return ResNet101(in_channels=cnn_in_channels, out_nodes=out_nodes)
    if cnn_arch == "resnet50":
        return ResNet50(in_channels=cnn_in_channels, out_nodes=out_nodes)
    if cnn_arch == "vgg16":
        return VGG16GLAM(in_channels=cnn_in_channels, out_nodes=out_nodes)
    if cnn_arch == "ViT":
        return ViT(
            img_size=img_size,
            patch_size=8,
            in_chans=cnn_in_channels,
            n_classes=out_nodes,
            p=0.1,
            attn_p=0.1,
        )
    raise ValueError("Invalid CNN architecture. Please choose from 'resnet101', 'resnet50', 'vgg16' or 'ViT'.")


def _build_temporal_encoder(
    rnn_arch: str,
    lstm_n_features: int,
    hidden_size: int,
    lstm_n_layers: int,
    lstm_out: int,
    seq_len: int,
):
    if rnn_arch == "LSTM":
        return rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
    if rnn_arch == "GRU":
        return rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
    if rnn_arch == "RNN":
        return rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
    if rnn_arch == "Transformer":
        return TSTransformerEncoderClassiregressor(
            feat_dim=lstm_n_features,
            max_len=seq_len,
            d_model=512,
            n_heads=8,
            num_layers=6,
            dim_feedforward=2048,
            num_classes=lstm_out,
            dropout=0.1,
            pos_encoding="fixed",
            activation="gelu",
            norm="BatchNorm",
            freeze=False,
        )
    raise ValueError("Invalid RNN architecture. Please choose from 'LSTM', 'GRU', 'RNN' or 'Transformer'.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing SoilNet...")
    x = torch.randn((4, 12, 64, 64), device=device)
    model = SoilNet(cnn_in_channels=12, cnn_arch="ViT").to(device)
    y = model(x)
    print(y.detach().shape)

    print("Testing SoilNetLSTM...")
    x_cnn = torch.randn((4, 12, 64, 64), device=device)
    x_lstm = torch.randn((4, 61, 10), device=device)
    model_lstm = SoilNetLSTM(
        cnn_arch="ViT",
        cnn_in_channels=12,
        regresor_input_from_cnn=1024,
        lstm_n_features=10,
        lstm_n_layers=2,
        lstm_out=128,
        hidden_size=128,
        rnn_arch="Transformer",
        seq_len=61,
    ).to(device)
    y_lstm = model_lstm((x_cnn, x_lstm))
    print(y_lstm.detach().shape)

    print("Testing SoilNetSimCLR (baseline)...")
    model_simclr = SoilNetSimCLR(
        cnn_arch="ViT",
        cnn_in_channels=12,
        regresor_input_from_cnn=128,
        lstm_n_features=10,
        lstm_n_layers=2,
        lstm_out=128,
        hidden_size=128,
        rnn_arch="Transformer",
        seq_len=61,
    ).to(device)
    z1, z2 = model_simclr((x_cnn, x_lstm))
    print(z1.detach().shape, z2.detach().shape)

    print("Testing SoilNetSimCLR (hybrid aux)...")
    model_hybrid = SoilNetSimCLR(
        cnn_arch="ViT",
        cnn_in_channels=12,
        regresor_input_from_cnn=128,
        lstm_n_features=10,
        lstm_n_layers=2,
        lstm_out=128,
        hidden_size=128,
        rnn_arch="Transformer",
        seq_len=61,
        use_hybrid_ssl=True,
    ).to(device)
    aux_out = model_hybrid((x_cnn, x_lstm), return_aux=True)
    print(aux_out["z_img"].shape, aux_out["z_clim"].shape, aux_out["clim_recon"].shape)
