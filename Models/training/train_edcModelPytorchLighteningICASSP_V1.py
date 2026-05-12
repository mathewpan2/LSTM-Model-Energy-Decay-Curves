"""
==========================================================
STFT Prediction with Optional Depth Map Features
Last updated: 2026-05-08
==========================================================
This script trains an LSTM model to predict STFT magnitude targets from room features.
It optionally fuses depth maps with the coordinate features during training and evaluation.
The model is implemented using PyTorch Lightning for streamlined training and evaluation.
Key Features:
- Target: Predicts flattened STFT magnitude spectra.
- Data Handling: Loads waveform metadata, computes STFT targets, and room features.
- Model Architecture: LSTM-based room encoder with optional depth-map encoder.
- Training: Includes early stopping and model checkpointing.
- Evaluation: Computes MAE and MSE, saves predictions and metadata.
- Visualization: Plots training and validation loss over time.
==========================================================
"""

import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import soundfile as sf
from scipy.signal import stft

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import importlib.util
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

_icassp_path = Path(__file__).resolve().parent / "icassp_lightning_components.py"
_icassp_spec = importlib.util.spec_from_file_location("icassp_lightning_components", _icassp_path)
_icassp_mod = importlib.util.module_from_spec(_icassp_spec)
assert _icassp_spec.loader is not None
_icassp_spec.loader.exec_module(_icassp_mod)
STFTModel = _icassp_mod.STFTModel

# ------------------------------
# Configuration
# ------------------------------
today = datetime.today().strftime('%Y-%m-%d')
resuts_path = os.path.join('Results/', today, time.strftime('%H-%M-%S'))
data_paths = os.path.join('Results/')
dataset_root = os.environ.get("ACOUSTIC_ROOMS_ROOT", "/mnt/code/code/AcousticRooms")
ir_base_path = os.path.join(dataset_root, "single_channel_ir")
metadata_path = os.path.join(dataset_root, "metadata")
depth_map_base_path = os.path.join(dataset_root, "depth_map")
model_save_dir = "Models"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(resuts_path, exist_ok=True)

usedMethod = "AcousticRooms_STFT"
waveform_length = 16000 # Samples per waveform used to build each STFT target
# rooms_to_process = 200 # Maximum number of rooms to keep (None for all)
rooms_to_process = None
# max_samples_to_load = 6000 # Total samples to load from AcousticRooms
max_samples_to_load = 10000
batch_size = 16 # Batch size for training
dataloader_num_workers = 4  # parallel workers for on-disk loading (set 0 if multiprocessing issues)
input_dim = 13  # coord encoder input after concatenating dist_m in forward()
coord_feature_dim = input_dim - 1  # scaled geometric feature vector from build_room_feature_row
N_REF_RIRS = 3
REF_RIR_FEAT_DIM = 6
REF_RIR_INPUT_DIM = N_REF_RIRS * REF_RIR_FEAT_DIM
REF_RIR_ENCODER_DIM = 64

# ============================================
# FLAG: Choose training target modality
# ============================================
USE_RAW_WAVEFORMS = True  # If True: train on raw waveforms; If False: train on STFT magnitude

# STFT parameters (only used when USE_RAW_WAVEFORMS=False)
stft_n_fft = 1024
stft_hop = 512
stft_time_frames = int(np.ceil(waveform_length / float(stft_hop)))
stft_freq_bins = stft_n_fft // 2 + 1
stft_target_shape = (stft_freq_bins, stft_time_frames)
stft_target_length = stft_freq_bins * stft_time_frames

# Raw waveform target (only used when USE_RAW_WAVEFORMS=True)
waveform_target_length = waveform_length
raw_waveform_early_reflection_ms = 80.0
raw_waveform_early_weight = 10.0
raw_waveform_late_weight = 1.0
raw_waveform_early_cutoff = None
reference_sample_rate = None

# Determine target length based on flag
target_length = waveform_target_length if USE_RAW_WAVEFORMS else stft_target_length

isScalingX = True
isScalingY = True
use_depth_map = True  # If True, fuse depth-map encoder features with coordinate features.
strict_depth_map = True  # If True, skip samples that are missing a matching depth map.
USE_REFERENCE_RIRS = True  # If False, dataset feeds zeros for ref features; model ignores ref input path.

# Data loading: False = preload all waveforms / depth / targets into RAM (faster epochs, high memory).
# True = load each sample from disk in __getitem__ (low memory, more disk I/O).
LAZY_LOADING = False

N_BANDS = 6
LATE_PARAMS = N_BANDS * 2  # T60 + energy per band (dropped onset delay for simplicity)
SAMPLE_RATE = 16000

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


def weighted_raw_waveform_loss(pred: torch.Tensor, target: torch.Tensor, early_cutoff: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if early_cutoff is None:
        raise ValueError("early_cutoff must be set for raw waveform training")

    early_cutoff = int(max(1, min(early_cutoff, pred.shape[1] - 1)))
    early_pred = pred[:, :early_cutoff]
    late_pred = pred[:, early_cutoff:]
    early_target = target[:, :early_cutoff]
    late_target = target[:, early_cutoff:]

    early_l1 = F.l1_loss(early_pred, early_target)
    late_l1 = F.l1_loss(late_pred, late_target)
    loss = raw_waveform_early_weight * early_l1 + raw_waveform_late_weight * late_l1
    return loss, early_l1, late_l1


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
    """
    Extract compact acoustic features from a single reference RIR.
    Returns a 6-dim feature vector.
    """
    rir = np.array(rir, dtype=np.float32)

    # --- T60 via linear regression on log energy decay ---
    window = sample_rate // 100  # 10ms windows
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

    # --- DRR ---
    peak_idx = int(np.abs(rir).argmax())
    direct_win = max(1, int(0.005 * sample_rate))  # 5ms
    direct_energy = float(np.sum(rir[peak_idx:peak_idx + direct_win]**2))
    reverb_energy = float(np.sum(rir[peak_idx + direct_win:]**2) + 1e-10)
    drr = float(10 * np.log10(direct_energy / reverb_energy + 1e-10))
    drr = float(np.clip(drr, -20, 20))

    # --- C50 ---
    peak_idx = int(np.abs(rir).argmax())
    early_cutoff = peak_idx + int(0.05 * sample_rate)
    early_e = np.sum(rir[peak_idx:early_cutoff]**2)
    late_e = np.sum(rir[early_cutoff:]**2) + 1e-10
    c50 = float(10 * np.log10(early_e / late_e))
    c50 = float(np.clip(c50, -20, 20))  

    # --- RMS ---
    rms = float(np.sqrt(np.mean(rir**2)))

    # --- Peak amplitude ---
    peak_amp = float(np.abs(rir).max())

    # --- EDT ---
    peak_idx = int(np.abs(rir).argmax())
    rir_from_peak = rir[peak_idx:]
    schroeder = np.cumsum(rir_from_peak[::-1] ** 2)[::-1]
    log_schroeder = 10 * np.log10(schroeder / (schroeder[0] + 1e-10) + 1e-10)
    t = np.arange(len(log_schroeder)) / sample_rate
    mask = log_schroeder >= -10.0
    if mask.sum() > 1:
        # interpolate to find exact -10dB crossing, extrapolate to 60dB
        t_cross = float(np.interp(-10.0, log_schroeder[mask][::-1], t[mask][::-1]))
        edt = float(np.clip(t_cross * 6.0, 0.0, 5.0))
    else:
        edt = t60  # fallback if decay is too fast to measure

    return np.array([t60, drr, c50, rms, peak_amp, edt], dtype=np.float32)



def load_preprocess_rir(rir_path: str, target_len: int) -> tuple[np.ndarray, int]:
    """Load WAV, delay-compensate (trim silent lead-in), trim/pad to target_len samples."""
    rir, sample_rate = sf.read(rir_path)
    sample_rate = int(sample_rate)
    if rir.ndim > 1:
        rir = rir[:, 0]
    rir = np.asarray(rir, dtype=np.float32).flatten()
    onset = int(np.abs(rir).argmax())
    lead_in = max(0, onset - 10)
    rir = rir[lead_in:]
    rir = rir[:target_len]
    if len(rir) < target_len:
        rir = np.pad(rir, (0, target_len - len(rir)), mode="constant")
    return np.asarray(rir, dtype=np.float32), sample_rate


def compute_geometric_distance_m(row: pd.Series) -> float:
    """Source–receiver distance (m), same geometry as build_room_feature_row."""
    return float(
        np.sqrt(
            (row["src_x"] - row["rec_x"]) ** 2
            + (row["src_y"] - row["rec_y"]) ** 2
            + (row["src_z"] - row["rec_z"]) ** 2
        )
    )


def build_room_feature_row(row: pd.Series, room_dims: dict, ref_sr: int) -> np.ndarray:
    """Scaled geometric feature row (distance excluded; passed separately as dist_m)."""
    L, W, H = room_dims[(row["room_type"], row["room_id"])]
    sx, sy, sz = float(row["src_x"]), float(row["src_y"]), float(row["src_z"])
    rx, ry, rz = float(row["rec_x"]), float(row["rec_y"]), float(row["rec_z"])
    dx, dy, dz = rx - sx, ry - sy, rz - sz
    return np.array(
        [sx, sy, sz, rx, ry, rz, L, W, H, dx, dy, dz],
        dtype=np.float32,
    )


def precompute_ref_scaled_by_room(
    ref_paths_by_room: dict,
    ref_min_arr: np.ndarray,
    ref_den_arr: np.ndarray,
    ref_sr: int,
) -> dict:
    """One-time ref-RIR features per room for DataLoader workers (shared, no per-worker cache)."""
    out = {}
    for rk, paths in ref_paths_by_room.items():
        feats = []
        for p in paths:
            if p is None:
                feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
            else:
                rir, sr = load_preprocess_rir(p, waveform_length)
                if sr != ref_sr:
                    raise ValueError(f"Sample rate mismatch for ref RIR {p}: expected {ref_sr}, got {sr}")
                feats.append(extract_rir_acoustic_features(rir, ref_sr))
        while len(feats) < N_REF_RIRS:
            feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
        arr = np.stack(feats, axis=0)
        scaled = (arr - ref_min_arr) / ref_den_arr
        out[rk] = scaled.astype(np.float32)
    return out


def build_stft_log_flat_target(rir: np.ndarray, sample_rate: int) -> np.ndarray:
    """Flattened log1p STFT magnitude matching training pipeline."""
    try:
        f, t_seg, Zxx = stft(
            rir,
            fs=sample_rate,
            nperseg=stft_n_fft,
            noverlap=stft_n_fft - stft_hop,
            boundary=None,
        )
        mag = np.abs(Zxx)
        if mag.shape[1] > stft_time_frames:
            mag = mag[:, :stft_time_frames]
        elif mag.shape[1] < stft_time_frames:
            mag = np.pad(mag, ((0, 0), (0, stft_time_frames - mag.shape[1])), mode="constant")
        mag = np.log1p(mag).astype(np.float32)
        return mag.reshape(-1).astype(np.float32)
    except Exception:
        mag = np.zeros((stft_freq_bins, stft_time_frames), dtype=np.float32)
        return mag.reshape(-1).astype(np.float32)


def load_depth_numpy(row: pd.Series, depth_base: str, strict: bool) -> np.ndarray:
    """Depth tensor shaped [1, H, W]."""
    receiver_match = re.search(r"_R(\d+)$", row["stem"])
    if receiver_match is None:
        raise ValueError(f"Cannot parse receiver index from stem: {row['stem']}")
    receiver_idx = int(receiver_match.group(1))
    depth_path = os.path.join(depth_base, row["room_type"], row["room_id"], f"{receiver_idx}.npy")
    if not os.path.exists(depth_path):
        if strict:
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        return np.zeros((1, 256, 512), dtype=np.float32)
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
    return depth_map.astype(np.float32)


def build_ref_paths_by_room(df: pd.DataFrame) -> dict:
    out = {}
    for (room_type, room_id), group in df.groupby(["room_type", "room_id"]):
        ref_rows = group.head(N_REF_RIRS)
        paths = [r["rir_path"] for _, r in ref_rows.iterrows()]
        while len(paths) < N_REF_RIRS:
            paths.append(None)
        out[(room_type, room_id)] = paths[:N_REF_RIRS]
    return out


def compute_ref_feature_bounds(ref_paths_by_room: dict, ref_sr: int, waveform_len: int) -> tuple[np.ndarray, np.ndarray]:
    stacks = []
    for paths in ref_paths_by_room.values():
        feats = []
        for p in paths:
            if p is None:
                feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
            else:
                rir, sr = load_preprocess_rir(p, waveform_len)
                if sr != ref_sr:
                    raise ValueError(f"Inconsistent sample rate for ref RIR {p}: expected {ref_sr}, got {sr}")
                feats.append(extract_rir_acoustic_features(rir, ref_sr))
        while len(feats) < N_REF_RIRS:
            feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
        stacks.append(np.stack(feats, axis=0))
    all_ref = np.stack(stacks, axis=0)
    ref_min = all_ref.min(axis=(0, 1), keepdims=True)
    ref_max = all_ref.max(axis=(0, 1), keepdims=True)
    ref_den = np.maximum(ref_max - ref_min, 1e-8)
    return ref_min, ref_den


def scan_depth_global_bounds(df: pd.DataFrame, depth_base: str, strict: bool) -> tuple[float, float]:
    dmin, dmax = np.inf, -np.inf
    for _, row in df.iterrows():
        try:
            dm = load_depth_numpy(row, depth_base, strict)
        except FileNotFoundError:
            continue
        dmin = min(dmin, float(dm.min()))
        dmax = max(dmax, float(dm.max()))
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        raise RuntimeError(
            "use_depth_map is True but no depth arrays could be loaded; check depth_map_base_path and manifest paths."
        )
    return float(dmin), float(dmax)


def stream_stft_target_bounds(df: pd.DataFrame, waveform_len: int) -> tuple[float, float]:
    gmin, gmax = np.inf, -np.inf
    for _, row in df.iterrows():
        try:
            rir, sr = load_preprocess_rir(row["rir_path"], waveform_len)
            flat = build_stft_log_flat_target(rir, sr)
            gmin = min(gmin, float(flat.min()))
            gmax = max(gmax, float(flat.max()))
        except Exception:
            continue
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise RuntimeError("STFT target bounds could not be computed (no readable RIR waveforms?).")
    return float(gmin), float(gmax)


dataset_df = load_acoustic_rooms_rows(ir_base_path, metadata_path)
if dataset_df.empty:
    raise RuntimeError(
        f"No AcousticRooms samples found under {ir_base_path} and {metadata_path}."
    )

dataset_df = dataset_df.sort_values(["room_type", "room_id", "stem"]).reset_index(drop=True)

# Compute room bounding box dimensions from aggregate positions
room_dimensions = compute_room_dimensions(dataset_df)

if rooms_to_process is not None:
    unique_rooms = dataset_df[["room_type", "room_id"]].drop_duplicates().reset_index(drop=True)
    if len(unique_rooms) > rooms_to_process:
        selected_rooms = unique_rooms.sample(n=rooms_to_process, random_state=42)
        room_keys = set(zip(selected_rooms["room_type"], selected_rooms["room_id"]))
        dataset_df = dataset_df[
            dataset_df.apply(lambda row: (row["room_type"], row["room_id"]) in room_keys, axis=1)
        ].reset_index(drop=True)

if max_samples_to_load is not None and len(dataset_df) > max_samples_to_load:
    dataset_df = dataset_df.sample(n=max_samples_to_load, random_state=42).sort_values(
        ["room_type", "room_id", "stem"]
    ).reset_index(drop=True)

print(f"\n{'='*60}")
print(f"Training Mode: {'RAW WAVEFORMS' if USE_RAW_WAVEFORMS else 'STFT MAGNITUDE'}")
print(f"Target Length: {target_length}")
print(
    f"Data: {'lazy load per batch (low RAM)' if LAZY_LOADING else 'preload full dataset into RAM'}"
)
print(f"{'='*60}\n")

if len(dataset_df) == 0:
    raise RuntimeError("Dataset manifest is empty.")

if LAZY_LOADING:
    _, reference_sample_rate = load_preprocess_rir(dataset_df.iloc[0]["rir_path"], waveform_length)
    reference_sample_rate = int(reference_sample_rate)

    room_feat_matrix = np.stack(
        [
            build_room_feature_row(dataset_df.iloc[i], room_dimensions, reference_sample_rate)
            for i in range(len(dataset_df))
        ],
        axis=0,
    )

    ref_paths_by_room = build_ref_paths_by_room(dataset_df)
    ref_min, ref_den = compute_ref_feature_bounds(ref_paths_by_room, reference_sample_rate, waveform_length)
    ref_scaled_by_room = precompute_ref_scaled_by_room(
        ref_paths_by_room, ref_min, ref_den, reference_sample_rate
    )
    if not USE_REFERENCE_RIRS:
        ref_scaled_by_room = {
            k: np.zeros((N_REF_RIRS, REF_RIR_FEAT_DIM), dtype=np.float32)
            for k in ref_scaled_by_room
        }

    if use_depth_map:
        depth_min, depth_max = scan_depth_global_bounds(dataset_df, depth_map_base_path, strict_depth_map)
        depth_den = max(depth_max - depth_min, 1e-8)
    else:
        depth_min = depth_max = depth_den = None

    if USE_RAW_WAVEFORMS:
        target_min = 0.0
        target_max = 1.0
        target_den = 1.0
    else:
        tg_min, tg_max = stream_stft_target_bounds(dataset_df, waveform_length)
        target_min = float(tg_min)
        target_max = float(tg_max)
        target_den = max(target_max - target_min, 1e-8)

    if USE_RAW_WAVEFORMS:
        raw_waveform_early_cutoff = int(round(reference_sample_rate * raw_waveform_early_reflection_ms / 1000.0))
        raw_waveform_early_cutoff = max(1, min(raw_waveform_early_cutoff, waveform_length - 1))
        print(
            f"Raw waveform early/late split: cutoff={raw_waveform_early_cutoff} samples "
            f"(~{raw_waveform_early_reflection_ms:.1f} ms at {reference_sample_rate} Hz)"
        )

    max_files_to_load = len(dataset_df)

    if USE_RAW_WAVEFORMS:
        scaler_save_name = "Models/scaler_waveform_acoustic_rooms.save"
        scaler_info = {
            "min": target_min,
            "max": target_max,
            "shape": (waveform_target_length,),
            "per_sample_peak_norm": True,
        }
    else:
        scaler_save_name = "Models/scaler_stft_acoustic_rooms.save"
        scaler_info = {"min": target_min, "max": target_max, "shape": stft_target_shape}

    joblib.dump(scaler_info, scaler_save_name)
    print(f"Saved target scaler to: {scaler_save_name}")

    if isScalingX:
        scaler_X = MinMaxScaler()
        scaler_X.fit(room_feat_matrix)
        joblib.dump(scaler_X, "Models/scaler_X_acoustic_rooms.save")
    else:
        scaler_X = None

else:
    # ----- Preload entire dataset (original behavior): high RAM, fast sampling -----
    all_targets = []
    room_features_list = []
    dist_m_list = []
    depth_maps_list = []
    loaded_rows = []
    depth_hw = None
    rir_cache = {}
    reference_sample_rate = None

    for row_idx, row in dataset_df.iterrows():
        try:
            rir, sample_rate = load_preprocess_rir(row["rir_path"], waveform_length)
            sample_rate = int(sample_rate)
            if reference_sample_rate is None:
                reference_sample_rate = sample_rate
            elif sample_rate != reference_sample_rate:
                raise ValueError(
                    f"Inconsistent sample rate: expected {reference_sample_rate}, got {sample_rate} for {row['rir_path']}"
                )

            rir_cache[(row["room_type"], row["room_id"], row["stem"])] = np.asarray(rir, dtype=np.float32).copy()

            if USE_RAW_WAVEFORMS:
                target = rir.astype(np.float32)
            else:
                target = build_stft_log_flat_target(rir, sample_rate)

            if use_depth_map:
                depth_map = load_depth_numpy(row, depth_map_base_path, strict_depth_map)
                if depth_hw is None:
                    depth_hw = tuple(depth_map.shape[-2:])
                elif tuple(depth_map.shape[-2:]) != depth_hw:
                    raise ValueError(
                        f"Inconsistent depth map shapes: expected {depth_hw}, got {tuple(depth_map.shape[-2:])}"
                    )
                depth_maps_list.append(depth_map)

            room_features_list.append(build_room_feature_row(row, room_dimensions, reference_sample_rate))
            dist_m_list.append(compute_geometric_distance_m(row))

            all_targets.append(target)
            loaded_rows.append(row.to_dict())
            print(f"Loaded {os.path.basename(row['rir_path'])} ({len(loaded_rows)}/{len(dataset_df)})")
        except Exception as e:
            print(f"Failed to load {row['rir_path']}: {e}")

    if not all_targets:
        raise RuntimeError("No AcousticRooms samples could be loaded successfully.")

    dataset_df = pd.DataFrame(loaded_rows).reset_index(drop=True)

    ref_rir_features_by_room = {}
    for (room_type, room_id), group in dataset_df.groupby(["room_type", "room_id"]):
        ref_rows = group.head(N_REF_RIRS)
        feats = []
        for _, ref_row in ref_rows.iterrows():
            key = (room_type, room_id, ref_row["stem"])
            if key in rir_cache:
                feats.append(extract_rir_acoustic_features(rir_cache[key], int(reference_sample_rate)))
            else:
                feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
        while len(feats) < N_REF_RIRS:
            feats.append(np.zeros(REF_RIR_FEAT_DIM, dtype=np.float32))
        ref_rir_features_by_room[(room_type, room_id)] = np.stack(feats).astype(np.float32)

    ref_rir_features = np.stack(
        [
            ref_rir_features_by_room[(r["room_type"], r["room_id"])]
            for r in loaded_rows
        ]
    ).astype(np.float32)

    if USE_RAW_WAVEFORMS:
        raw_waveform_early_cutoff = int(round(reference_sample_rate * raw_waveform_early_reflection_ms / 1000.0))
        raw_waveform_early_cutoff = max(1, min(raw_waveform_early_cutoff, waveform_length - 1))
        print(
            f"Raw waveform early/late split: cutoff={raw_waveform_early_cutoff} samples "
            f"(~{raw_waveform_early_reflection_ms:.1f} ms at {reference_sample_rate} Hz)"
        )

    combined_data = np.stack(all_targets).astype(np.float32)
    room_features_arr = np.stack(room_features_list, axis=0).astype(np.float32)
    dist_m_arr = np.asarray(dist_m_list, dtype=np.float32).reshape(-1, 1)

    depth_maps = np.stack(depth_maps_list).astype(np.float32) if use_depth_map else None
    max_files_to_load = len(dataset_df)

    if use_depth_map and depth_maps is not None:
        depth_min = float(np.min(depth_maps))
        depth_max = float(np.max(depth_maps))
        depth_den = max(depth_max - depth_min, 1e-8)
        depth_maps = (depth_maps - depth_min) / depth_den
    else:
        depth_min = depth_max = None

    if USE_RAW_WAVEFORMS:
        peaks = np.abs(combined_data).max(axis=1, keepdims=True).clip(min=1e-8)
        targets_scaled = combined_data / peaks
        target_min = 0.0
        target_max = 1.0
        target_den = 1.0
    else:
        target_min = float(np.min(combined_data))
        target_max = float(np.max(combined_data))
        target_den = max(target_max - target_min, 1e-8)
        targets_scaled = (combined_data - target_min) / target_den

    if USE_RAW_WAVEFORMS:
        scaler_save_name = "Models/scaler_waveform_acoustic_rooms.save"
        scaler_info = {
            "min": target_min,
            "max": target_max,
            "shape": (waveform_target_length,),
            "per_sample_peak_norm": True,
        }
    else:
        scaler_save_name = "Models/scaler_stft_acoustic_rooms.save"
        scaler_info = {"min": target_min, "max": target_max, "shape": stft_target_shape}

    joblib.dump(scaler_info, scaler_save_name)
    print(f"Saved target scaler to: {scaler_save_name}")

    if isScalingX:
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(room_features_arr)
        X_scaled = X_scaled.reshape((-1, 1, coord_feature_dim))
        joblib.dump(scaler_X, "Models/scaler_X_acoustic_rooms.save")
    else:
        scaler_X = None
        X_scaled = room_features_arr.reshape((-1, 1, coord_feature_dim))

    ref_min = ref_rir_features.min(axis=(0, 1), keepdims=True)
    ref_max = ref_rir_features.max(axis=(0, 1), keepdims=True)
    ref_den = np.maximum(ref_max - ref_min, 1e-8)
    ref_rir_features_scaled = (ref_rir_features - ref_min) / ref_den
    if not USE_REFERENCE_RIRS:
        ref_rir_features_scaled = np.zeros_like(ref_rir_features_scaled)

    if isScalingY:
        y_scaled = targets_scaled
    else:
        y_scaled = combined_data

    if USE_RAW_WAVEFORMS:
        peak_positions = np.abs(y_scaled).argmax(axis=1)
        print(
            f"Peak sample stats: min={int(peak_positions.min())}, "
            f"max={int(peak_positions.max())}, mean={float(peak_positions.mean()):.0f}"
        )


class STFTDataset(Dataset):
    """Loads RIR and depth from disk per sample; ref-RIR features come from precomputed per-room tables."""

    def __init__(
        self,
        sample_indices: np.ndarray,
        meta_df: pd.DataFrame,
        *,
        room_dims: dict,
        ref_sr: int,
        scaler_x,
        scale_x: bool,
        ref_scaled_by_room: dict,
        use_depth: bool,
        depth_base: str,
        strict_depth: bool,
        dmin: float | None,
        dmax: float | None,
        use_raw: bool,
        t_min: float,
        t_den: float,
        scale_y: bool,
    ):
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.meta_df = meta_df.reset_index(drop=True)
        self.room_dims = room_dims
        self.ref_sr = int(ref_sr)
        self.scaler_x = scaler_x
        self.scale_x = scale_x
        self.ref_scaled_by_room = ref_scaled_by_room
        self.use_depth = use_depth
        self.depth_base = depth_base
        self.strict_depth = strict_depth
        self.dmin = dmin
        self.dmax = dmax
        self.use_raw = use_raw
        self.t_min = t_min
        self.t_den = t_den
        self.scale_y = scale_y

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        j = int(self.sample_indices[i])
        row = self.meta_df.iloc[j]
        rir, sr = load_preprocess_rir(row["rir_path"], waveform_length)
        if sr != self.ref_sr:
            raise ValueError(f"Sample rate mismatch for {row['rir_path']}: expected {self.ref_sr}, got {sr}")

        if self.use_raw:
            if self.scale_y:
                y = rir.astype(np.float32)
                y = y / (np.abs(y).max() + 1e-8)
            else:
                y = rir.astype(np.float32)
        else:
            flat = build_stft_log_flat_target(rir, sr)
            if self.scale_y:
                y = (flat - self.t_min) / self.t_den
            else:
                y = flat.astype(np.float32)

        rf = build_room_feature_row(row, self.room_dims, self.ref_sr)
        if self.scale_x and self.scaler_x is not None:
            X = self.scaler_x.transform(rf.reshape(1, -1)).astype(np.float32).reshape(1, 1, coord_feature_dim)
        else:
            X = rf.reshape(1, 1, coord_feature_dim).astype(np.float32)

        rk = (row["room_type"], row["room_id"])
        ref_s = np.asarray(self.ref_scaled_by_room[rk], dtype=np.float32).reshape(N_REF_RIRS, REF_RIR_FEAT_DIM)
        dist_m = torch.tensor([compute_geometric_distance_m(row)], dtype=torch.float32)

        if self.use_depth:
            d_raw = load_depth_numpy(row, self.depth_base, self.strict_depth)
            d_norm = (d_raw - self.dmin) / max(self.dmax - self.dmin, 1e-8)
            return (
                torch.from_numpy(X).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(ref_s).float(),
                torch.from_numpy(d_norm).float(),
                dist_m,
            )
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(ref_s).float(),
            dist_m,
        )


class STFTDatasetPreloaded(Dataset):
    """In-memory tensors built during startup (LAZY_LOADING=False)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ref_rirs: np.ndarray,
        depth: np.ndarray | None = None,
        dist_m: np.ndarray | None = None,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.ref_rirs = torch.as_tensor(ref_rirs, dtype=torch.float32)
        self.depth = torch.as_tensor(depth, dtype=torch.float32) if depth is not None else None
        self.dist_m = torch.as_tensor(dist_m, dtype=torch.float32) if dist_m is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.depth is not None:
            return self.X[idx], self.y[idx], self.ref_rirs[idx], self.depth[idx], self.dist_m[idx]
        return self.X[idx], self.y[idx], self.ref_rirs[idx], self.dist_m[idx]


if LAZY_LOADING:
    rng_indices = np.arange(len(dataset_df))
    idx_temp, idx_test = train_test_split(rng_indices, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.25, random_state=42)

    _ds_kw = dict(
        meta_df=dataset_df,
        room_dims=room_dimensions,
        ref_sr=reference_sample_rate,
        scaler_x=scaler_X,
        scale_x=isScalingX,
        ref_scaled_by_room=ref_scaled_by_room,
        use_depth=use_depth_map,
        depth_base=depth_map_base_path,
        strict_depth=strict_depth_map,
        dmin=depth_min,
        dmax=depth_max,
        use_raw=USE_RAW_WAVEFORMS,
        t_min=target_min,
        t_den=target_den,
        scale_y=isScalingY,
    )

    train_dataset = STFTDataset(idx_train, **_ds_kw)
    val_dataset = STFTDataset(idx_val, **_ds_kw)
    test_dataset = STFTDataset(idx_test, **_ds_kw)
else:
    if use_depth_map and depth_maps is not None:
        X_temp, X_test, R_temp, R_test, D_temp, D_test, y_temp, y_test, dm_temp, dm_test = train_test_split(
            X_scaled,
            ref_rir_features_scaled,
            depth_maps,
            y_scaled,
            dist_m_arr,
            test_size=0.2,
            random_state=42,
        )
        X_train, X_val, R_train, R_val, D_train, D_val, y_train, y_val, dm_train, dm_val = train_test_split(
            X_temp,
            R_temp,
            D_temp,
            y_temp,
            dm_temp,
            test_size=0.25,
            random_state=42,
        )
    else:
        X_temp, X_test, R_temp, R_test, y_temp, y_test, dm_temp, dm_test = train_test_split(
            X_scaled,
            ref_rir_features_scaled,
            y_scaled,
            dist_m_arr,
            test_size=0.2,
            random_state=42,
        )
        X_train, X_val, R_train, R_val, y_train, y_val, dm_train, dm_val = train_test_split(
            X_temp,
            R_temp,
            y_temp,
            dm_temp,
            test_size=0.25,
            random_state=42,
        )
        D_train = D_val = D_test = None

    train_dataset = STFTDatasetPreloaded(X_train, y_train, R_train, depth=D_train, dist_m=dm_train)
    val_dataset = STFTDatasetPreloaded(X_val, y_val, R_val, depth=D_val, dist_m=dm_val)
    test_dataset = STFTDatasetPreloaded(X_test, y_test, R_test, depth=D_test, dist_m=dm_test)

_pin = torch.cuda.is_available()
_eff_workers = dataloader_num_workers
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_eff_workers,
    pin_memory=_pin,
    persistent_workers=_eff_workers > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_eff_workers,
    pin_memory=_pin,
    persistent_workers=_eff_workers > 0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_eff_workers,
    pin_memory=_pin,
    persistent_workers=_eff_workers > 0,
)

# STFTModel and related layers are loaded from icassp_lightning_components.py (see imports above).

# ------------------------------
# Logging and Training
# ------------------------------
log_dir = f"{resuts_path}/lightning_logs"
logger = TensorBoardLogger(save_dir=log_dir, name=f"stft_model_{max_files_to_load}_{stft_target_length}")

model = STFTModel(
    input_dim=input_dim,
    target_length=target_length,
    use_depth_map=use_depth_map,
    early_cutoff_samples=raw_waveform_early_cutoff if USE_RAW_WAVEFORMS else None,
    physics_sample_rate=float(reference_sample_rate),
    use_reference_rirs=USE_REFERENCE_RIRS,
)
print(f"early_weight: {raw_waveform_early_weight}")
print(f"late_weight: {raw_waveform_late_weight}")

early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=model_save_dir, filename="best_model", save_top_k=1)

maxEpochs = 200
trainer = pl.Trainer(
    max_epochs=maxEpochs,
    callbacks=[early_stop, checkpoint],
    accelerator='auto',
    devices='auto',
    log_every_n_steps=5,
    enable_progress_bar=True,
    enable_model_summary=True,
    gradient_clip_val=1.0,
)

trainer.fit(model, train_loader, val_loader)
print(f"Model saved at: {checkpoint.best_model_path}")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(model.epoch_train_loss_history, label='Train Loss')
plt.plot(model.epoch_val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss (per epoch)')
plt.legend()
plt.grid(True)
plt.savefig(f"{resuts_path}/loss_plot.png")
plt.show()

# ------------------------------
# Evaluate
# ------------------------------
model.eval()
preds, targets = [], []
for batch in val_loader:
    if use_depth_map:
        X, y, ref_rirs, depth, dist_m = batch
    else:
        X, y, ref_rirs, dist_m = batch
        depth = None

    with torch.no_grad():
        dev = model.device
        if depth is not None:
            output = model(
                X.to(dev),
                ref_rirs.to(dev),
                depth.to(dev),
                dist_m=dist_m.to(dev),
            )
        else:
            output = model(X.to(dev), ref_rirs.to(dev), dist_m=dist_m.to(dev))
    preds.append(output.cpu().numpy())
    targets.append(y.numpy())

preds = np.vstack(preds)
targets = np.vstack(targets)

if isScalingY:
    preds_rescaled = preds * target_den + target_min
    targets_rescaled = targets * target_den + target_min
else:
    preds_rescaled = preds
    targets_rescaled = targets

mae = mean_absolute_error(targets_rescaled, preds_rescaled)
mse = mean_squared_error(targets_rescaled, preds_rescaled)
print(f"Test MAE: {mae:.5f}, MSE: {mse:.5f}")

# Save outputs
np.save(f"{resuts_path}/predicted_stft_sample_{max_files_to_load}_{stft_target_length}.npy", preds)
np.save(f"{resuts_path}/actual_stft_sample_{max_files_to_load}_{stft_target_length}.npy", targets)
np.save(f"{resuts_path}/predicted_stft_sample_rescaled_{max_files_to_load}_{stft_target_length}.npy", preds_rescaled)
np.save(f"{resuts_path}/actual_stft_sample_rescaled_{max_files_to_load}_{stft_target_length}.npy", targets_rescaled)

# Metadata
metadata = {
    "waveform_length": waveform_length,
    "stft_target_length": stft_target_length,
    "use_raw_waveforms": bool(USE_RAW_WAVEFORMS),
    "raw_waveform_early_reflection_ms": raw_waveform_early_reflection_ms if USE_RAW_WAVEFORMS else None,
    "raw_waveform_early_cutoff": raw_waveform_early_cutoff if USE_RAW_WAVEFORMS else None,
    "raw_waveform_early_weight": raw_waveform_early_weight if USE_RAW_WAVEFORMS else None,
    "raw_waveform_late_weight": raw_waveform_late_weight if USE_RAW_WAVEFORMS else None,
    "reference_sample_rate": reference_sample_rate if USE_RAW_WAVEFORMS else None,
    "dataset_root": dataset_root,
    "rooms_to_process": rooms_to_process,
    "samples_loaded": max_files_to_load,
    "batch_size": batch_size,
    "input_dim": input_dim,
    "loss": "MSE",
    "maxEpochs": maxEpochs,
    "mae": float(mae),
    "mse": float(mse),
    "model_save_path": str(checkpoint.best_model_path),
    "scaler_X": bool(isScalingX),
    "scaler_y": bool(isScalingY),
    "use_depth_map": bool(use_depth_map),
    "depth_map_min": depth_min,
    "depth_map_max": depth_max,
    "method": usedMethod,
    "LAZY_LOADING": bool(LAZY_LOADING),
    "Training Dataset Size": len(train_dataset),
    "Validation Dataset Size": len(val_dataset),
    "Test Dataset Size": len(test_dataset),
    "dataloader_num_workers": dataloader_num_workers,
    "analytical_direct_sound": True,
}

json_path = os.path.join(resuts_path, "experiment_metadata.json")
with open(json_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {json_path}")

print("Training and evaluation completed successfully!")
