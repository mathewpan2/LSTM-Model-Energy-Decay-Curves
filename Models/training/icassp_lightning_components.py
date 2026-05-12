"""
Shared Lightning model and helpers for ICASSP AcousticRooms training.
Imported by train_edcModelPytorchLighteningICASSP_V1.py and notebooks — no dataset I/O here.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

N_REF_RIRS = 3
REF_RIR_FEAT_DIM = 6
REF_RIR_ENCODER_DIM = 64
SAMPLE_RATE = 16000


class DepthResBlock(nn.Module):
    """Residual block with skip connection for depth encoder."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class RefRIREncoder(nn.Module):
    """Encodes N reference RIR feature vectors into a single room embedding."""

    def __init__(self, feat_dim=REF_RIR_FEAT_DIM, n_refs=N_REF_RIRS, output_dim=REF_RIR_ENCODER_DIM):
        super().__init__()
        self.per_rir_encoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.aggregator = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, ref_feats):
        encoded = self.per_rir_encoder(ref_feats)
        pooled = encoded.mean(dim=1)
        return self.aggregator(pooled)


def multiscale_stft_loss(pred, target, fft_sizes=(32, 64, 128, 256)):
    """L1 on magnitude STFTs at multiple resolutions. pred/target: (B, T)."""
    max_fft = max(fft_sizes)
    if pred.shape[-1] < max_fft:
        pad_len = max_fft - pred.shape[-1]
        pred = F.pad(pred, (0, pad_len))
        target = F.pad(target, (0, pad_len))
    loss = pred.new_tensor(0.0)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        p = torch.stft(
            pred,
            n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        t = torch.stft(
            target,
            n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        loss = loss + F.l1_loss(p, t)
    return loss / len(fft_sizes)


def analytical_late_tail(features, late_length, sample_rate=SAMPLE_RATE):
    """Physics-inspired fallback late tail synthesis."""
    B = features.shape[0]
    device = features.device
    t60 = torch.full((B, 1), 0.7, device=device)
    t = torch.arange(late_length, device=device).float().unsqueeze(0) / sample_rate
    decay = torch.exp(-6.9078 * t / t60)
    noise = torch.randn(B, late_length, device=device)
    noise = noise - noise.mean(dim=1, keepdim=True)
    return noise * decay * 0.05


class STFTModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        target_length,
        use_depth_map=False,
        depth_encoder_dim=128,
        early_cutoff_samples=None,
        physics_sample_rate: float = 16000.0,
        use_reference_rirs: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_depth_map = bool(use_depth_map)
        self.use_reference_rirs = bool(use_reference_rirs)
        self.early_cutoff_samples = early_cutoff_samples
        self.physics_sample_rate = float(physics_sample_rate)
        self.train_losses = []
        self.val_losses = []
        self.epoch_train_loss_history: list[float] = []
        self.epoch_val_loss_history: list[float] = []
        self._printed_alignment_debug = False

        self.coord_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )

        coord_dim = 128
        self.ref_rir_encoder = RefRIREncoder(
            feat_dim=REF_RIR_FEAT_DIM,
            n_refs=N_REF_RIRS,
            output_dim=REF_RIR_ENCODER_DIM,
        )

        if self.use_depth_map:
            self.depth_encoder = nn.Sequential(
                DepthResBlock(1, 16),
                DepthResBlock(16, 32),
                DepthResBlock(32, 64),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, depth_encoder_dim),
                nn.ReLU(inplace=True),
            )
            fused_dim = coord_dim + REF_RIR_ENCODER_DIM + depth_encoder_dim
        else:
            fused_dim = coord_dim + REF_RIR_ENCODER_DIM

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        fused_head_dim = 256

        if early_cutoff_samples is not None and early_cutoff_samples > 0:
            self.early_head = nn.Sequential(
                nn.Linear(fused_head_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, early_cutoff_samples),
            )
            late_length = max(1, target_length - early_cutoff_samples)
        else:
            self.early_head = None
            late_length = target_length

        self.late_length = late_length

    def forward(self, x, ref_rirs, depth=None, dist_m: torch.Tensor | None = None):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if dist_m is not None:
            d = dist_m.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, d], dim=1)

        if ref_rirs.dim() == 4 and ref_rirs.size(1) == 1:
            ref_rirs = ref_rirs.squeeze(1)

        coord_features = self.coord_encoder(x)
        if not self.use_reference_rirs:
            ref_rirs = torch.zeros_like(ref_rirs)
        ref_features = self.ref_rir_encoder(ref_rirs)

        if coord_features.dim() > 2:
            coord_features = coord_features.reshape(coord_features.size(0), -1)
        if ref_features.dim() > 2:
            ref_features = ref_features.reshape(ref_features.size(0), -1)

        if self.use_depth_map:
            if depth is None:
                raise ValueError("Depth map input is required when use_depth_map=True")
            depth_features = self.depth_encoder(depth)
            if depth_features.dim() > 2:
                depth_features = depth_features.reshape(depth_features.size(0), -1)
            features = torch.cat([coord_features, ref_features, depth_features], dim=1)
        else:
            features = torch.cat([coord_features, ref_features], dim=1)

        fused = self.fusion(features)

        if self.early_head is not None:
            early_pred = self.early_head(fused)
            late_pred = analytical_late_tail(fused, self.late_length, self.physics_sample_rate)
            output = torch.cat([early_pred, late_pred], dim=1)
            expected_len = int(self.early_cutoff_samples + self.late_length)
            if output.shape[1] != expected_len:
                raise RuntimeError(f"Early/late concat length mismatch: got {output.shape[1]}, expected {expected_len}")
        else:
            output = analytical_late_tail(fused, self.late_length, self.physics_sample_rate)

        return output

    def training_step(self, batch, batch_idx):
        if self.use_depth_map:
            X, y, ref_rirs, depth, dist_m = batch
            y_hat = self(X, ref_rirs, depth, dist_m=dist_m)
        else:
            X, y, ref_rirs, dist_m = batch
            y_hat = self(X, ref_rirs, dist_m=dist_m)

        if not self._printed_alignment_debug and y.shape[0] > 0:
            peak_idx = int(y[0].abs().argmax().item())
            print(f"debug_rir_peak_index_first_batch: {peak_idx}")
            self._printed_alignment_debug = True

        y_norm = y
        y_hat_norm = y_hat

        early_cutoff = self.early_cutoff_samples
        if early_cutoff is not None and early_cutoff > 0:
            y_early = y_norm[:, :early_cutoff]
            y_hat_early = y_hat_norm[:, :early_cutoff]
            early_l1 = F.l1_loss(y_hat_early, y_early)
            early_stft = multiscale_stft_loss(y_hat_early, y_early)
            early_loss = (early_l1 * 5.0) + (early_stft * 5.0)
            if batch_idx == 0 and self.current_epoch < 3:
                print(f"y_hat_early mean abs: {y_hat_early.abs().mean().item():.6f}")
                print(f"y_early mean abs: {y_early.abs().mean().item():.6f}")
                print(f"early_l1: {early_l1.item():.6f}  early_stft: {early_stft.item():.6f}")
                print(f"early_loss (weighted sum): {early_loss.item():.6f}")
            late_loss = torch.zeros_like(early_loss)
            loss = early_loss
            self.log("train_early_l1", early_l1, on_step=True, on_epoch=True)
            self.log("train_early_stft", early_stft, on_step=True, on_epoch=True)
        else:
            early_loss = F.l1_loss(y_hat_norm, y_norm)
            late_loss = torch.zeros_like(early_loss)
            loss = early_loss
        mae = torch.mean(torch.abs(y_hat_norm - y_norm))
        self.train_losses.append(loss.detach())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_early_loss", early_loss, on_step=True, on_epoch=True)
        self.log("train_late_loss", late_loss, on_step=True, on_epoch=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_losses:
            avg_train_loss = torch.stack(self.train_losses).mean()
            self.epoch_train_loss_history.append(float(avg_train_loss.detach().cpu()))
            self.log("train_loss_epoch", avg_train_loss, prog_bar=True)
            self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        if self.use_depth_map:
            X, y, ref_rirs, depth, dist_m = batch
            y_hat = self(X, ref_rirs, depth, dist_m=dist_m)
        else:
            X, y, ref_rirs, dist_m = batch
            y_hat = self(X, ref_rirs, dist_m=dist_m)

        y_norm = y
        y_hat_norm = y_hat

        early_cutoff = self.early_cutoff_samples
        if early_cutoff is not None and early_cutoff > 0:
            y_early = y_norm[:, :early_cutoff]
            y_hat_early = y_hat_norm[:, :early_cutoff]
            early_l1 = F.l1_loss(y_hat_early, y_early)
            early_stft = multiscale_stft_loss(y_hat_early, y_early)
            early_loss = (early_l1 * 5.0) + (early_stft * 5.0)
            late_loss = torch.zeros_like(early_loss)
            loss = early_loss
        else:
            early_loss = F.l1_loss(y_hat_norm, y_norm)
            late_loss = torch.zeros_like(early_loss)
            loss = early_loss
        mae = torch.mean(torch.abs(y_hat_norm - y_norm))
        self.val_losses.append(loss.detach())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_early_loss", early_loss, on_epoch=True)
        self.log("val_late_loss", late_loss, on_epoch=True)
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.epoch_val_loss_history.append(float(avg_val_loss.detach().cpu()))
            self.log("val_loss_epoch", avg_val_loss, prog_bar=True)
            self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
