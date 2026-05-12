"""
==========================================================================
 STFT Magnitude Inference using an LSTM-based Room Model
==========================================================================

Author: Imran Muhammad
Date: 2026-05-08

📌 DESCRIPTION
-------------
This script performs inference using a pre-trained LSTM model that predicts
flattened STFT magnitude targets from room configuration features.

The model and scalers are trained separately. For inference, you have two options:

1️⃣ **Option 1 - Use Existing Dataset**
    - Select a random example from a pre-generated dataset.
    - Actual STFT magnitude will be computed and compared to the predicted STFT.
    - Plots will show both actual and predicted results.

2️⃣ **Option 2 - Use Custom Room Features**
    - Manually enter or use default room dimensions and positions.
    - Only predicted STFT magnitude will be generated.

📌 INPUT FEATURES FORMAT (9 features)
-------------------------------------
[src_x, src_y, src_z, rec_x, rec_y, rec_z, room_length, room_width, room_height]

📌 OUTPUT
--------
- Predicted STFT magnitude spectrum
- Optional comparison with actual dataset (if using Option 1)
- If `clap22.wav` is found (or path in env `CLAP22_W`), convolves that dry sound with GT and predicted RIRs and saves listenable WAVs under `inference_results/`.

📌 USAGE
-------
$ python inference_edcModelPytorchLighteningV3.py

Make sure:
- `Models/` folder contains:
     - `best_model.ckpt`
     - `scaler_X_acoustic_rooms.save`
     - `scaler_stft_acoustic_rooms.save`
- AcousticRooms is available at `/mnt/code/code/AcousticRooms` or via `ACOUSTIC_ROOMS_ROOT`.

==========================================================================
"""

import os
import json
import re
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import soundfile as sf
from scipy.signal import stft, istft, fftconvolve, resample

# ==========================================================
#  Model Definition (same as used in training)
# ==========================================================

SAMPLE_RATE = 16000
N_REF_RIRS = 3
REF_RIR_FEAT_DIM = 6
REF_RIR_ENCODER_DIM = 64


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
    """Encodes reference RIR acoustic descriptors into one room embedding."""
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


def analytical_late_tail(features, late_length, sample_rate=SAMPLE_RATE):
    B = features.shape[0]
    device = features.device
    t60 = torch.full((B, 1), 0.4, device=device)
    t = torch.arange(late_length, device=device).float().unsqueeze(0) / sample_rate
    decay = torch.exp(-6.9078 * t / t60)
    noise = torch.randn(B, late_length, device=device)
    noise = noise - noise.mean(dim=1, keepdim=True)
    return noise * decay * 0.01


class STFTModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        target_length,
        use_depth_map=False,
        depth_encoder_dim=128,
        early_cutoff_samples=None,
        physics_sample_rate: float = 16000.0,
    ):
        super().__init__()
        self.use_depth_map = bool(use_depth_map)
        self.early_cutoff_samples = early_cutoff_samples
        self.physics_sample_rate = float(physics_sample_rate)
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
            self.late_length = max(1, target_length - early_cutoff_samples)
        else:
            self.early_head = None
            self.late_length = target_length

    def forward(self, x, ref_rirs, depth=None, dist_m: torch.Tensor | None = None):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if dist_m is not None:
            d = dist_m.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, d], dim=1)
        if ref_rirs.dim() == 4 and ref_rirs.size(1) == 1:
            ref_rirs = ref_rirs.squeeze(1)

        coord_features = self.coord_encoder(x)
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
            late_pred = analytical_late_tail(fused, self.late_length)
            output = torch.cat([early_pred, late_pred], dim=1)
            expected_len = int(self.early_cutoff_samples + self.late_length)
            if output.shape[1] != expected_len:
                raise RuntimeError(f"Early/late concat length mismatch: got {output.shape[1]}, expected {expected_len}")
        else:
            output = analytical_late_tail(fused, self.late_length)
        return output

def compute_stft_magnitude(rir: np.ndarray, sample_rate: int, n_fft: int, hop: int, target_shape: tuple[int, int]) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32).flatten()
    rir = rir[: target_shape[1] * hop]
    if len(rir) < target_shape[1] * hop:
        rir = np.pad(rir, (0, target_shape[1] * hop - len(rir)), mode="constant")

    _, _, zxx = stft(rir, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag = np.abs(zxx).astype(np.float32)
    if mag.shape[0] > target_shape[0]:
        mag = mag[: target_shape[0], :]
    elif mag.shape[0] < target_shape[0]:
        mag = np.pad(mag, ((0, target_shape[0] - mag.shape[0]), (0, 0)), mode="constant")
    if mag.shape[1] > target_shape[1]:
        mag = mag[:, : target_shape[1]]
    elif mag.shape[1] < target_shape[1]:
        mag = np.pad(mag, ((0, 0), (0, target_shape[1] - mag.shape[1])), mode="constant")
    return np.log1p(mag).astype(np.float32)


def normalize_stft(mag: np.ndarray, scaler_info: dict) -> np.ndarray:
    min_value = float(scaler_info["min"])
    max_value = float(scaler_info["max"])
    return (mag - min_value) / max(max_value - min_value, 1e-8)


def denormalize_stft(mag_scaled: np.ndarray, scaler_info: dict) -> np.ndarray:
    min_value = float(scaler_info["min"])
    max_value = float(scaler_info["max"])
    return mag_scaled * max(max_value - min_value, 1e-8) + min_value


def reconstruct_waveform_from_stft(stft_mag: np.ndarray, sample_rate: int, n_fft: int, hop: int) -> np.ndarray:
    """Reconstruct waveform from STFT magnitude using Griffin-Lim phase reconstruction."""
    # Use Griffin-Lim algorithm to estimate phase
    phase = np.random.rand(stft_mag.shape[0], stft_mag.shape[1]) * 2 * np.pi
    stft_complex = stft_mag * np.exp(1j * phase)
    
    # Run Griffin-Lim iterations to refine phase
    noverlap = n_fft - hop
    target_shape = stft_mag.shape
    
    for _ in range(10):  # 10 iterations of Griffin-Lim
        waveform = istft(stft_complex, fs=sample_rate, nperseg=n_fft, noverlap=noverlap)[1]
        _, _, stft_recon = stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=noverlap, boundary=None)
        
        # Pad or trim the reconstructed STFT to match target shape
        if stft_recon.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - stft_recon.shape[1]
            stft_recon = np.pad(stft_recon, ((0, 0), (0, pad_width)), mode='constant')
        elif stft_recon.shape[1] > target_shape[1]:
            stft_recon = stft_recon[:, :target_shape[1]]
        
        stft_complex = stft_mag * np.exp(1j * np.angle(stft_recon))
    
    waveform = istft(stft_complex, fs=sample_rate, nperseg=n_fft, noverlap=noverlap)[1]
    return waveform.astype(np.float32)


def resolve_clap22_path(search_name: str = "clap22.wav") -> str | None:
    """Find dry source WAV: env CLAP22_WAV, then script dir, cwd, inference_results."""
    if p := os.environ.get("CLAP22_WAV", "").strip():
        if os.path.isfile(p):
            return p
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.getcwd(), os.path.join(here, "inference_results")):
        cand = os.path.join(d, search_name)
        if os.path.isfile(cand):
            return cand
    return None


def load_dry_sound(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Mono float32 dry signal at target_sr."""
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = np.mean(x.astype(np.float32), axis=1)
    else:
        x = np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).flatten()
    sr = int(sr)
    if sr != target_sr and len(x) > 1:
        n_out = max(1, int(round(len(x) * target_sr / sr)))
        x = resample(x, n_out).astype(np.float32)
        sr = target_sr
    return x, sr


def convolve_source_with_rir(dry: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """FFT convolution dry * rir (same units as numpy convolve full). Outputs float32."""
    dry_f = np.asarray(dry, dtype=np.float64)
    rir_f = np.asarray(rir, dtype=np.float64)
    out = fftconvolve(dry_f, rir_f, mode="full")
    return out.astype(np.float32)


def peak_normalize_for_listen(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.abs(x).max())
    return (x / max(m, 1e-12) * peak).astype(np.float32)


def save_auralizations(
    out_dir: str,
    basename: str,
    dry_path: str,
    *,
    rir_gt: np.ndarray | None,
    rir_pred: np.ndarray,
    sample_rate: int,
):
    """Write convolved listenable WAVs. If rir_gt is None, only predicted."""
    rir_gt = None if rir_gt is None else np.asarray(rir_gt, dtype=np.float32).flatten()
    rir_pred = np.asarray(rir_pred, dtype=np.float32).flatten()

    dry, sr_d = load_dry_sound(dry_path, sample_rate)
    if sr_d != sample_rate:
        raise RuntimeError(f"dry sample rate {sr_d} != {sample_rate} after load_dry_sound")

    base = os.path.join(out_dir, basename)
    if rir_gt is not None and len(rir_gt) > 0:
        y_gt = convolve_source_with_rir(dry, rir_gt)
        y_gt = peak_normalize_for_listen(y_gt)
        path_gt = f"{base}_clap_gt_rir.wav"
        sf.write(path_gt, y_gt, sample_rate)
        print(f"Saved dry @ {dry_path} convolved with ground-truth RIR → {path_gt}")

    if len(rir_pred) > 0:
        y_pr = convolve_source_with_rir(dry, rir_pred)
        y_pr = peak_normalize_for_listen(y_pr)
        path_pr = f"{base}_clap_pred_rir.wav"
        sf.write(path_pr, y_pr, sample_rate)
        print(f"Saved dry @ {dry_path} convolved with predicted RIR → {path_pr}")


def load_acoustic_rooms_rows(ir_base: str, meta_base: str) -> pd.DataFrame:
    rows = []

    for room_type in sorted(os.listdir(ir_base)):
        room_type_path = os.path.join(ir_base, room_type)
        if not os.path.isdir(room_type_path):
            continue

        for room_id in sorted(os.listdir(room_type_path)):
            room_path = os.path.join(room_type_path, room_id)
            if not os.path.isdir(room_path):
                continue

            for file_name in sorted(os.listdir(room_path)):
                if not file_name.endswith("_hybrid_IR.wav"):
                    continue

                stem = file_name.replace("_hybrid_IR.wav", "")
                rir_path = os.path.join(room_path, file_name)
                json_path = os.path.join(meta_base, room_type, room_id, f"{stem}.json")
                if not os.path.exists(json_path):
                    continue

                with open(json_path, "r") as f:
                    meta = json.load(f)

                rows.append({
                    "room_type": room_type,
                    "room_id": room_id,
                    "stem": stem,
                    "rir_path": rir_path,
                    "src_x": meta["src_loc"][0],
                    "src_y": meta["src_loc"][1],
                    "src_z": meta["src_loc"][2],
                    "rec_x": meta["rec_loc"][0],
                    "rec_y": meta["rec_loc"][1],
                    "rec_z": meta["rec_loc"][2],
                })

    return pd.DataFrame(rows)


def compute_room_dimensions(df: pd.DataFrame) -> dict:
    """Compute room bounding box dimensions (L, W, H) for each room from aggregate source/receiver positions."""
    room_dims = {}
    for (room_type, room_id), group in df.groupby(["room_type", "room_id"]):
        xs = np.concatenate([group["src_x"].values, group["rec_x"].values])
        ys = np.concatenate([group["src_y"].values, group["rec_y"].values])
        zs = np.concatenate([group["src_z"].values, group["rec_z"].values])
        
        length = float(np.max(xs) - np.min(xs))  # X-axis
        width = float(np.max(ys) - np.min(ys))   # Y-axis
        height = float(np.max(zs) - np.min(zs))  # Z-axis
        
        room_dims[(room_type, room_id)] = (length, width, height)
    return room_dims


def extract_rir_acoustic_features(rir, sample_rate=16000):
    """Extract compact acoustic descriptors from one reference RIR."""
    rir = np.array(rir, dtype=np.float32)

    window = sample_rate // 100
    n_windows = len(rir) // window
    if n_windows < 5:
        t60 = 0.4
    else:
        windows = rir[:n_windows * window].reshape(n_windows, window)
        energy = np.mean(windows**2, axis=1)
        peak_e = np.max(energy) + 1e-10
        log_e = np.log(energy / peak_e + 1e-10)
        t = np.arange(n_windows) * (window / sample_rate)
        slope = np.polyfit(t, log_e, 1)[0]
        t60 = float(np.clip(-6.9078 / (slope - 1e-10), 0.1, 5.0))

    peak_idx = int(np.abs(rir).argmax())
    direct_win = max(1, int(0.005 * sample_rate))
    direct_energy = float(np.sum(rir[peak_idx:peak_idx + direct_win]**2))
    reverb_energy = float(np.sum(rir[peak_idx + direct_win:]**2) + 1e-10)
    drr = float(10 * np.log10(direct_energy / reverb_energy + 1e-10))
    drr = float(np.clip(drr, -20, 20))

    early_cutoff = int(0.05 * sample_rate)
    early_energy = float(np.sum(rir[:early_cutoff]**2))
    total_energy = float(np.sum(rir**2) + 1e-10)
    early_ratio = float(np.clip(early_energy / total_energy, 0, 1))

    rms = float(np.sqrt(np.mean(rir**2)))
    peak_amp = float(np.abs(rir).max())
    energy_per_sample = rir**2
    centroid = float(np.sum(np.arange(len(rir)) * energy_per_sample) / (np.sum(energy_per_sample) + 1e-10)) / sample_rate
    centroid = float(np.clip(centroid, 0, 1))

    return np.array([t60, drr, early_ratio, rms, peak_amp, centroid], dtype=np.float32)


def build_ref_rir_feature_lookup(df: pd.DataFrame, waveform_length: int):
    """Build per-room (3, 6) ref-RIR features and global scaling stats."""
    ref_by_room = {}
    all_refs = []
    for (room_type, room_id), group in df.groupby(["room_type", "room_id"]):
        ref_rows = group.head(N_REF_RIRS)
        feats = []
        for _, ref_row in ref_rows.iterrows():
            rir_ref, sr_ref = sf.read(ref_row["rir_path"])
            sr_ref = int(sr_ref)
            if rir_ref.ndim > 1:
                rir_ref = rir_ref[:, 0]
            rir_ref = np.asarray(rir_ref, dtype=np.float32).flatten()
            onset_ref = int(np.abs(rir_ref).argmax())
            lead_in_ref = max(0, onset_ref - 10)
            rir_ref = rir_ref[lead_in_ref:]
            rir_ref = rir_ref[:waveform_length]
            if len(rir_ref) < waveform_length:
                rir_ref = np.pad(rir_ref, (0, waveform_length - len(rir_ref)), mode="constant")
            feats.append(extract_rir_acoustic_features(rir_ref, sr_ref))
        while len(feats) < N_REF_RIRS:
            feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
        room_feats = np.stack(feats).astype(np.float32)
        ref_by_room[(room_type, room_id)] = room_feats
        all_refs.append(room_feats)

    all_refs = np.stack(all_refs).astype(np.float32)
    ref_min = all_refs.min(axis=(0, 1), keepdims=True)
    ref_max = all_refs.max(axis=(0, 1), keepdims=True)
    ref_den = np.maximum(ref_max - ref_min, 1e-8)

    scaled_lookup = {
        key: ((value - ref_min[0]) / ref_den[0]).astype(np.float32)
        for key, value in ref_by_room.items()
    }
    return scaled_lookup


def infer_model_config_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    has_depth_encoder = any("depth_encoder" in key for key in state_dict.keys())
    has_early_head = any("early_head" in key for key in state_dict.keys())
    depth_encoder_dim = 128
    early_cutoff_samples = None

    for key, value in state_dict.items():
        if key.endswith("depth_encoder.7.weight") and len(value.shape) == 2:
            depth_encoder_dim = int(value.shape[0])
        if key.endswith("early_head.5.bias") and len(value.shape) == 1:
            early_cutoff_samples = int(value.shape[0])

    if not has_early_head:
        early_cutoff_samples = None

    return has_depth_encoder, depth_encoder_dim, early_cutoff_samples


def load_model_from_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    target_length: int,
    use_depth_map: bool,
    physics_sample_rate: float = 16000.0,
):
    has_depth_encoder, depth_encoder_dim, early_cutoff_samples = infer_model_config_from_checkpoint(checkpoint_path)
    model = STFTModel(
        input_dim=input_dim,
        target_length=target_length,
        use_depth_map=use_depth_map and has_depth_encoder,
        depth_encoder_dim=depth_encoder_dim,
        early_cutoff_samples=early_cutoff_samples,
        physics_sample_rate=physics_sample_rate,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    compatible_state_dict = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state_dict[key] = value
        else:
            skipped_keys.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing checkpoint keys ignored: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected checkpoint keys ignored: {unexpected_keys}")
    if skipped_keys:
        print(f"Warning: skipped incompatible checkpoint keys: {skipped_keys}")

    if use_depth_map and not has_depth_encoder:
        print("Warning: depth maps were requested, but this checkpoint has no depth encoder. Running without depth input.")
        model.use_depth_map = False

    if int(target_length) != int(model.late_length + (model.early_cutoff_samples or 0)):
        print(
            "Warning: checkpoint reconstruction length differs from scaler target length. "
            "The model may run, but predictions could be shape-mismatched."
        )

    return model


def load_depth_map_for_row(depth_base: str, row: pd.Series) -> np.ndarray:
    receiver_match = re.search(r"_R(\d+)$", row["stem"])
    if receiver_match is None:
        raise ValueError(f"Cannot parse receiver index from stem: {row['stem']}")

    receiver_idx = int(receiver_match.group(1))
    depth_path = os.path.join(depth_base, row["room_type"], row["room_id"], f"{receiver_idx}.npy")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    depth_raw = np.load(depth_path)
    depth_raw = np.asarray(depth_raw, dtype=np.float32)

    if depth_raw.ndim == 2:
        depth_map = depth_raw[np.newaxis, :, :]
    elif depth_raw.ndim == 3:
        if depth_raw.shape[0] in (1, 3):
            depth_map = depth_raw[:1, :, :]
        elif depth_raw.shape[-1] in (1, 3):
            depth_map = np.transpose(depth_raw, (2, 0, 1))[:1, :, :]
        else:
            raise ValueError(f"Unsupported depth map shape: {depth_raw.shape}")
    else:
        raise ValueError(f"Unsupported depth map ndim: {depth_raw.ndim}")

    # Apply per-map min-max normalization so value range matches training convention.
    dmin = float(np.min(depth_map))
    dmax = float(np.max(depth_map))
    depth_map = (depth_map - dmin) / max(dmax - dmin, 1e-8)
    return depth_map.astype(np.float32)


def predict_stft(
    model,
    scaler_y,
    features_scaled: np.ndarray,
    ref_rirs_scaled: np.ndarray,
    depth_map: np.ndarray = None,
    dist_m: float | None = None,
) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        ref_tensor = torch.tensor(ref_rirs_scaled[np.newaxis, ...], dtype=torch.float32)
        dist_tensor = (
            torch.tensor([[dist_m]], dtype=torch.float32) if dist_m is not None else None
        )
        if model.use_depth_map:
            if depth_map is None:
                raise ValueError("Depth map is required for this checkpoint.")
            depth_tensor = torch.tensor(depth_map[np.newaxis, ...], dtype=torch.float32)
            pred_scaled = model(x_tensor, ref_tensor, depth_tensor, dist_m=dist_tensor)
        else:
            pred_scaled = model(x_tensor, ref_tensor, dist_m=dist_tensor)

    pred_scaled = pred_scaled.cpu().numpy()
    if scaler_y.get("per_sample_peak_norm", False):
        pred_flat = pred_scaled[0]
    else:
        pred_flat = denormalize_stft(pred_scaled[0], scaler_y)
    return pred_flat.reshape(tuple(scaler_y.get("shape", (pred_flat.shape[0], 1))))

# ==========================================================
#  MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    # ============================================
    # FLAG: Choose inference target modality
    # ============================================
    USE_RAW_WAVEFORMS = True # Must match training flag! If True: raw waveforms; If False: STFT magnitude

    # ------------------------------
    # Configuration
    # ------------------------------
    FS = 16000
    waveform_length = 16000
    stft_n_fft = 1024
    stft_hop = 512
    stft_time_frames = int(np.ceil(waveform_length / float(stft_hop)))
    stft_freq_bins = stft_n_fft // 2 + 1
    default_stft_shape = (stft_freq_bins, stft_time_frames)
    input_dim = 13  # coord features(12) + dist_m concatenated in forward
    coord_feature_dim = input_dim - 1
    
    print(f"\n{'='*60}")
    print(f"Inference Mode: {'RAW WAVEFORMS' if USE_RAW_WAVEFORMS else 'STFT MAGNITUDE'}")
    print(f"{'='*60}\n")

    dataset_root = os.environ.get("ACOUSTIC_ROOMS_ROOT", "/mnt/code/code/AcousticRooms")
    ir_base_path = os.path.join(dataset_root, "single_channel_ir")
    metadata_path = os.path.join(dataset_root, "metadata")
    depth_map_base_path = os.path.join(dataset_root, "depth_map")
    dataset_df = load_acoustic_rooms_rows(ir_base_path, metadata_path)
    if dataset_df.empty:
        raise RuntimeError(
            f"No AcousticRooms samples found under {ir_base_path} and {metadata_path}."
        )

    # Recreate the same train/test split as in training to avoid data leakage
    indices = np.arange(len(dataset_df))
    temp_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_val_indices, _ = train_test_split(temp_indices, test_size=0.25, random_state=42)
    
    # For inference, use test set to avoid data leakage
    test_df = dataset_df.iloc[test_indices].reset_index(drop=True)
    room_dimensions = compute_room_dimensions(dataset_df)
    ref_rir_lookup_scaled = build_ref_rir_feature_lookup(dataset_df, waveform_length)
    print(f"Dataset info: Total samples={len(dataset_df)}, Test set samples={len(test_df)}")


    checkpoint_path = "Models/best_model-v18.ckpt"

    scaler_X_path = "Models/scaler_X_acoustic_rooms.save"
    
    # Load appropriate scaler based on mode
    if USE_RAW_WAVEFORMS:
        scaler_y_path = "Models/scaler_waveform_acoustic_rooms.save"
    else:
        scaler_y_path = "Models/scaler_stft_acoustic_rooms.save"

    use_depth_map_ckpt, depth_encoder_dim_ckpt, early_cutoff_ckpt = infer_model_config_from_checkpoint(checkpoint_path)
    print(
        f"Checkpoint model config: use_depth_map={use_depth_map_ckpt}, "
        f"depth_encoder_dim={depth_encoder_dim_ckpt}, "
        f"early_cutoff_samples={early_cutoff_ckpt}"
    )

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Determine target shape/length based on mode
    if USE_RAW_WAVEFORMS:
        stft_target_shape = (waveform_length,)
        stft_target_length = waveform_length
    else:
        stft_target_shape = tuple(scaler_y.get("shape", default_stft_shape))
        stft_target_length = int(np.prod(stft_target_shape))
    print("Use depth maps when available? (y/n)")
    use_depth_map_choice = input("Enter choice (y/n): ").strip().lower() == "y"

    model = load_model_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        target_length=stft_target_length,
        use_depth_map=use_depth_map_choice,
        physics_sample_rate=float(FS),
    )
    model.eval()

    # Create output folder
    os.makedirs("inference_results", exist_ok=True)

    clap_path = resolve_clap22_path()
    if clap_path:
        print(f"Dry stimulus for convolution (set CLAP22_WAV to override): {clap_path}")
    else:
        print(
            "clap22.wav not found (searched env CLAP22_WAV, script dir, cwd, inference_results/). "
            "Auralization WAVs will be skipped unless you add the file."
        )

    print("\n==============================")
    print("  STFT Magnitude Inference")
    print("==============================")
    print("Select mode:")
    print("1 - Use existing dataset example")
    print("2 - Use custom room features")
    print("3 - Batch evaluation on 10 test samples")
    choice = input("Enter choice (1/2/3): ").strip()

    # ==========================================================
    #  OPTION 1: Existing Dataset
    # ==========================================================
    if choice == "1":
        print("\nYou selected: Use existing dataset")

        rng = np.random.default_rng()
        rand_idx = int(rng.integers(0, len(test_df)))
        selected_row = test_df.iloc[rand_idx]
        selected_features = selected_row[["src_x", "src_y", "src_z", "rec_x", "rec_y", "rec_z"]].to_numpy(dtype=np.float32)

        print(f"Selected room (from TEST set): {selected_row['room_type']} / {selected_row['room_id']} / {selected_row['stem']}")

        rir, _ = sf.read(selected_row["rir_path"])
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = np.asarray(rir, dtype=np.float32).flatten()

        # Load actual target based on mode
        if USE_RAW_WAVEFORMS:
            # For raw waveforms: apply the same per-sample peak normalization as training.
            actual_target = rir[:waveform_length]
            if len(actual_target) < waveform_length:
                actual_target = np.pad(actual_target, (0, waveform_length - len(actual_target)), mode="constant")
            actual_peak = np.abs(actual_target).max()
            actual_target = actual_target / max(actual_peak, 1e-8)
            actual_stft = None  # Not used in waveform mode
        else:
            # For STFT: compute the STFT magnitude
            actual_stft = compute_stft_magnitude(rir, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop, target_shape=stft_target_shape)
            actual_target = None  # Not directly used

        # Room bbox + displacement (rec - src) + distance (m); matches training scaler
        L, W, H = room_dimensions[(selected_row["room_type"], selected_row["room_id"])]
        sx, sy, sz = selected_row["src_x"], selected_row["src_y"], selected_row["src_z"]
        rx, ry, rz = selected_row["rec_x"], selected_row["rec_y"], selected_row["rec_z"]
        dist = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
        dx, dy, dz = float(rx - sx), float(ry - sy), float(rz - sz)
        selected_features_with_dims = np.concatenate([selected_features, [L, W, H, dx, dy, dz]])
        
        # Scale input features
        new_features_scaled = scaler_X.transform(selected_features_with_dims.reshape(1, -1))
        new_features_scaled = new_features_scaled.reshape(1, 1, coord_feature_dim)

        depth_map = None
        if model.use_depth_map:
            depth_map = load_depth_map_for_row(depth_map_base_path, selected_row)
        room_key = (selected_row["room_type"], selected_row["room_id"])
        ref_rirs_scaled = ref_rir_lookup_scaled[room_key]

        pred_target = predict_stft(model, scaler_y, new_features_scaled, ref_rirs_scaled, depth_map, dist_m=dist)

        # Compute metrics based on mode
        if USE_RAW_WAVEFORMS:
            # For raw waveforms: direct comparison
            actual_flat = actual_target.reshape(-1)
            pred_flat = pred_target.reshape(-1)
            metric_type = "Waveform"
        else:
            # For STFT: compare magnitude spectrograms
            actual_stft = actual_target if actual_target is not None else compute_stft_magnitude(rir, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop, target_shape=stft_target_shape)
            pred_stft = pred_target
            actual_flat = actual_stft.reshape(-1)
            pred_flat = pred_stft.reshape(-1)
            metric_type = "STFT"
        
        mse = mean_squared_error(actual_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_flat, pred_flat)
        r2 = r2_score(actual_flat, pred_flat)
        pearson_corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
        
        print(f"{metric_type} MSE  = {mse:.6f}")
        print(f"{metric_type} RMSE = {rmse:.6f}")
        print(f"{metric_type} MAE  = {mae:.6f}")
        print(f"R² Score = {r2:.6f}")
        print(f"Pearson Correlation = {pearson_corr:.6f}")

        # Prepare waveforms for visualization
        if USE_RAW_WAVEFORMS:
            # Mode: raw waveforms (direct targets)
            actual_waveform = actual_target
            pred_waveform = pred_target
            rir_trimmed = rir
        else:
            # Mode: STFT (reconstruct waveforms via ISTFT)
            actual_stft = actual_target if actual_target is not None else compute_stft_magnitude(rir, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop, target_shape=stft_target_shape)
            pred_stft = pred_target
            actual_waveform = reconstruct_waveform_from_stft(actual_stft, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop)
            pred_waveform = reconstruct_waveform_from_stft(pred_stft, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop)
            rir_trimmed = rir
        
        # Trim waveforms to match for fair comparison
        min_len = min(len(actual_waveform), len(pred_waveform), len(rir_trimmed))
        actual_waveform = actual_waveform[:min_len]
        pred_waveform = pred_waveform[:min_len]
        rir_trimmed = rir_trimmed[:min_len]

        # Plot results
        time_axis = np.arange(min_len) / FS
        
        # Determine number of subplots based on mode
        n_plots = 5 if not USE_RAW_WAVEFORMS else 3
        fig, axs = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots))
        
        plot_idx = 0
        
        # STFT plots (only in STFT mode)
        if not USE_RAW_WAVEFORMS:
            actual_stft_plot = actual_stft if 'actual_stft' in locals() else compute_stft_magnitude(rir, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop, target_shape=stft_target_shape)
            pred_stft_plot = pred_target
            diff_stft = np.abs(actual_stft_plot - pred_stft_plot)
            
            im0 = axs[plot_idx].imshow(actual_stft_plot, aspect="auto", origin="lower")
            axs[plot_idx].set_title("Actual STFT Magnitude")
            fig.colorbar(im0, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
            
            im1 = axs[plot_idx].imshow(pred_stft_plot, aspect="auto", origin="lower")
            axs[plot_idx].set_title("Predicted STFT Magnitude")
            fig.colorbar(im1, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
            
            im2 = axs[plot_idx].imshow(diff_stft, aspect="auto", origin="lower")
            axs[plot_idx].set_title("STFT Absolute Difference")
            fig.colorbar(im2, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
        
        # Waveform plots (all modes)
        axs[plot_idx].plot(time_axis, rir_trimmed, label="Original RIR", linewidth=0.8, alpha=0.7)
        axs[plot_idx].plot(time_axis, actual_waveform, label="Actual", linewidth=0.8, alpha=0.7)
        if not USE_RAW_WAVEFORMS:
            axs[plot_idx].set_title("Original RIR vs Reconstructed from Actual STFT")
        else:
            axs[plot_idx].set_title("Original RIR vs Actual Waveform")
        axs[plot_idx].set_xlabel("Time (s)")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        axs[plot_idx].plot(time_axis, actual_waveform, label="Actual", linewidth=0.8, alpha=0.7)
        axs[plot_idx].plot(time_axis, pred_waveform, label="Predicted", linewidth=0.8, alpha=0.7)
        axs[plot_idx].set_title("Waveform Comparison: Actual vs Predicted")
        axs[plot_idx].set_xlabel("Time (s)")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, alpha=0.3)

        metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nCorr: {pearson_corr:.4f}"
        axs[0].text(0.98, 0.97, metrics_text,
                    transform=axs[0].transAxes, verticalalignment='top',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

        plt.tight_layout()
        mode_str = "waveform" if USE_RAW_WAVEFORMS else "stft"
        plt.savefig(f"inference_results/comparison_existing_{mode_str}.png", dpi=300)
        plt.show()

        if clap_path:
            stem_safe = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in str(selected_row["stem"])
            )
            save_auralizations(
                "inference_results",
                f"aural_{stem_safe}_{mode_str}",
                clap_path,
                rir_gt=actual_waveform,
                rir_pred=pred_waveform,
                sample_rate=FS,
            )

    # ==========================================================
    #  OPTION 2: Custom Room Features
    # ==========================================================
    elif choice == "2":
        print("\nYou selected: Use custom room features")
        use_defaults = input("Use default example values? (y/n): ").strip().lower()

        if use_defaults == "y":
            src = [0.9136, -2.399, 1.9912]
            rec = [-1.3396, -2.4405, 0.9182]
        else:
            src = [float(input(f"Source {axis} (m): ")) for axis in ['X','Y','Z']]
            rec = [float(input(f"Receiver {axis} (m): ")) for axis in ['X','Y','Z']]

        selected_features = np.array(src + rec)
        
        # For custom mode, ask user for room dimensions or use defaults
        use_default_dims = input("Use default room dimensions? (y/n): ").strip().lower()
        if use_default_dims == "y":
            L, W, H = 5.0, 4.0, 2.5  # Default room dimensions (meters)
            print(f"Using default room dimensions: L={L}m, W={W}m, H={H}m")
        else:
            L = float(input("Room length L (m): "))
            W = float(input("Room width W (m): "))
            H = float(input("Room height H (m): "))
        
        sx, sy, sz = selected_features[0], selected_features[1], selected_features[2]
        rx, ry, rz = selected_features[3], selected_features[4], selected_features[5]
        dist = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
        dx, dy, dz = float(rx - sx), float(ry - sy), float(rz - sz)
        selected_features_with_dims = np.concatenate([selected_features, [L, W, H, dx, dy, dz]])
        print(f"Custom feature vector (6 coords + 3 dims + disp3): {selected_features_with_dims}")

        # Predict only (no actual)
        new_features_scaled = scaler_X.transform(selected_features_with_dims.reshape(1, -1)).reshape(1, 1, coord_feature_dim)

        depth_map = None
        if model.use_depth_map:
            # Custom mode has no linked AcousticRooms sample; use neutral depth input.
            depth_map = np.zeros((1, 256, 512), dtype=np.float32)
            print("Checkpoint expects depth map. Using zero depth map in custom mode.")
        ref_rirs_scaled = np.zeros((N_REF_RIRS, REF_RIR_FEAT_DIM), dtype=np.float32)

        pred_target = predict_stft(model, scaler_y, new_features_scaled, ref_rirs_scaled, depth_map, dist_m=dist)

        # Prepare output based on mode
        if USE_RAW_WAVEFORMS:
            pred_waveform = pred_target
            time_axis = np.arange(len(pred_waveform)) / FS
            
            # Plot predicted waveform only
            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            ax.plot(time_axis, pred_waveform, linewidth=0.8)
            ax.set_title("Predicted Waveform")
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
        else:
            pred_stft = pred_target
            pred_waveform = reconstruct_waveform_from_stft(pred_stft, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop)
            time_axis = np.arange(len(pred_waveform)) / FS

            # Plot predicted STFT and waveform
            fig, axs = plt.subplots(2, 1, figsize=(14, 10))
            im = axs[0].imshow(pred_stft, aspect="auto", origin="lower")
            axs[0].set_title("Predicted STFT Magnitude")
            fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

            axs[1].plot(time_axis, pred_waveform, linewidth=0.8)
            axs[1].set_title("Reconstructed Waveform from Predicted STFT")
            axs[1].set_xlabel("Time (s)")
            axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        mode_str = "waveform" if USE_RAW_WAVEFORMS else "stft"
        plt.savefig(f"inference_results/predicted_only_custom_{mode_str}.png", dpi=300)
        plt.show()

        if clap_path:
            save_auralizations(
                "inference_results",
                f"aural_custom_{mode_str}",
                clap_path,
                rir_gt=None,
                rir_pred=pred_waveform,
                sample_rate=FS,
            )

    # ==========================================================
    #  OPTION 3: Batch Evaluation
    # ==========================================================
    elif choice == "3":
        print("\nYou selected: Batch evaluation on test set")
        num_samples = min(10, len(test_df))
        
        all_mse = []
        all_rmse = []
        all_mae = []
        all_r2 = []
        all_corr = []
        
        print(f"\nEvaluating {num_samples} test samples...")
        for i in range(num_samples):
            selected_row = test_df.iloc[i]
            selected_features = selected_row[["src_x", "src_y", "src_z", "rec_x", "rec_y", "rec_z"]].to_numpy(dtype=np.float32)
            
            # Load actual target based on mode
            rir, _ = sf.read(selected_row["rir_path"])
            if rir.ndim > 1:
                rir = rir[:, 0]
            rir = np.asarray(rir, dtype=np.float32).flatten()
            
            if USE_RAW_WAVEFORMS:
                actual_target = rir[:waveform_length]
                if len(actual_target) < waveform_length:
                    actual_target = np.pad(actual_target, (0, waveform_length - len(actual_target)), mode="constant")
                actual_peak = np.abs(actual_target).max()
                actual_target = actual_target / max(actual_peak, 1e-8)
            else:
                actual_target = compute_stft_magnitude(rir, sample_rate=FS, n_fft=stft_n_fft, hop=stft_hop, target_shape=stft_target_shape)
            
            # Room bbox + displacement + distance (matches training)
            L, W, H = room_dimensions[(selected_row["room_type"], selected_row["room_id"])]
            sx, sy, sz = selected_row["src_x"], selected_row["src_y"], selected_row["src_z"]
            rx, ry, rz = selected_row["rec_x"], selected_row["rec_y"], selected_row["rec_z"]
            dist = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
            dx, dy, dz = float(rx - sx), float(ry - sy), float(rz - sz)
            selected_features_with_dims = np.concatenate([selected_features, [L, W, H, dx, dy, dz]])
            
            # Predict
            new_features_scaled = scaler_X.transform(selected_features_with_dims.reshape(1, -1))
            new_features_scaled = new_features_scaled.reshape(1, 1, coord_feature_dim)
            
            depth_map = None
            if model.use_depth_map:
                depth_map = load_depth_map_for_row(depth_map_base_path, selected_row)
            room_key = (selected_row["room_type"], selected_row["room_id"])
            ref_rirs_scaled = ref_rir_lookup_scaled[room_key]

            pred_target = predict_stft(model, scaler_y, new_features_scaled, ref_rirs_scaled, depth_map, dist_m=dist)

            if clap_path and i == 0:
                if USE_RAW_WAVEFORMS:
                    w_gt = actual_target.astype(np.float32).flatten()
                    w_pred = pred_target.astype(np.float32).flatten()
                else:
                    w_gt = reconstruct_waveform_from_stft(actual_target, FS, stft_n_fft, stft_hop)
                    w_pred = reconstruct_waveform_from_stft(pred_target, FS, stft_n_fft, stft_hop)
                lw = min(len(w_gt), len(w_pred))
                w_gt = w_gt[:lw]
                w_pred = w_pred[:lw]
                mode_str_b = "waveform" if USE_RAW_WAVEFORMS else "stft"
                save_auralizations(
                    "inference_results",
                    f"aural_batch_sample0_{mode_str_b}",
                    clap_path,
                    rir_gt=w_gt,
                    rir_pred=w_pred,
                    sample_rate=FS,
                )

            # Compute metrics
            actual_flat = actual_target.reshape(-1)
            pred_flat = pred_target.reshape(-1)
            mse = mean_squared_error(actual_flat, pred_flat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_flat, pred_flat)
            r2 = r2_score(actual_flat, pred_flat)
            corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
            
            all_mse.append(mse)
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_r2.append(r2)
            all_corr.append(corr)
            
            print(f"  Sample {i+1}/{num_samples}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"MSE:  Mean={np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
        print(f"RMSE: Mean={np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
        print(f"MAE:  Mean={np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        print(f"R²:   Mean={np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}  (Range: {np.min(all_r2):.4f} - {np.max(all_r2):.4f})")
        print(f"Corr: Mean={np.mean(all_corr):.4f} ± {np.std(all_corr):.4f}")
        print("="*60)
    
    else:
        print("❌ Invalid choice. Please run the script again.")
