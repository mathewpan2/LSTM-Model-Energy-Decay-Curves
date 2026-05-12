"""
==========================================================================
 Room Impulse Response (RIR) Reconstruction using LSTM-based EDC Prediction
==========================================================================

Author: Imran Muhammad
Date: 2025-09-27

📌 DESCRIPTION
-------------
This script performs inference using a pre-trained LSTM model that predicts
Energy Decay Curves (EDCs) from room configuration features. It then reconstructs
the corresponding Room Impulse Responses (RIRs) using the Random Sign-Sticky method.

The model and scalers are trained separately. For inference, you have two options:

1️⃣ **Option 1 - Use Existing Dataset**  
   - Select a random example from a pre-generated dataset.  
   - Actual EDC will be loaded and compared to the predicted EDC.  
   - Plots will show both actual and predicted results.

2️⃣ **Option 2 - Use Custom Room Features**  
   - Manually enter or use default room dimensions, positions, and absorption.  
   - Only predicted EDC, RIR, and FFT will be generated (no ground truth).

📌 INPUT FEATURES FORMAT (16 features)
--------------------------------------
[L, W, H, src_x, src_y, src_z, rec_x, rec_y, rec_z, absorption_band1..7]

📌 OUTPUT
--------
- Predicted EDC curve (dB)
- Predicted RIR waveform
- FFT magnitude response
- Optional comparison with actual dataset (if using Option 1)

📌 USAGE
-------
$ python inference_edcModelPytorchLighteningV3.py

Make sure:
- `Models/` folder contains:
    - `best_model.ckpt`
    - `scaler_X_*.save`
    - `scaler_edc_*.save`
- Dataset CSV and EDC .npy files are in correct paths if using Option 1.

==========================================================================
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

import soundfile as sf

from scipy.signal import fftconvolve
from scipy.signal import stft

# ==========================================================
#  Model Definition (same as used in training)
# ==========================================================

# STFT params aligned with training
train_stft_n_fft = 1024
train_stft_hop = 512
train_stft_time_frames = int(np.ceil(16000 / float(train_stft_hop)))
train_stft_freq_bins = train_stft_n_fft // 2 + 1


def compute_stft_np(wave: np.ndarray, sample_rate: int):
    try:
        f, t_seg, Zxx = stft(wave, fs=sample_rate, nperseg=train_stft_n_fft, noverlap=train_stft_n_fft - train_stft_hop, boundary=None)
        mag = np.abs(Zxx)
        if mag.shape[1] > train_stft_time_frames:
            mag = mag[:, :train_stft_time_frames]
        elif mag.shape[1] < train_stft_time_frames:
            pad_w = train_stft_time_frames - mag.shape[1]
            mag = np.pad(mag, ((0, 0), (0, pad_w)), mode='constant')
        return np.log1p(mag).astype(np.float32)
    except Exception:
        return np.zeros((train_stft_freq_bins, train_stft_time_frames), dtype=np.float32)


class EDCModel(pl.LightningModule):
    def __init__(self, input_dim, target_length, depth_enabled=False, depth_encoder_dim=128, stft_encoder_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.depth_enabled = depth_enabled
        if self.depth_enabled:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, depth_encoder_dim),
                nn.ReLU(inplace=True),
            )

        self.stft_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, stft_encoder_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = 128 + stft_encoder_dim
        if self.depth_enabled:
            fused_dim += depth_encoder_dim

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(fused_dim, 2048)
        self.fc2 = nn.Linear(2048, target_length)

    def forward(self, x, depth=None):
        x = x.to(self.device)
        _, (h_n, _) = self.lstm(x)
        coord_features = h_n[-1]

        depth_features = None
        if self.depth_enabled:
            if depth is None:
                raise ValueError("Depth map required for this model")
            depth_features = self.depth_encoder(depth.to(self.device))

        stft_tmp = getattr(self, '_stft_tmp', None)
        if stft_tmp is not None:
            stft_input = stft_tmp
            if stft_input.dim() == 3:
                stft_input = stft_input.unsqueeze(1)
            stft_features = self.stft_encoder(stft_input.to(self.device))
        else:
            stft_features = torch.zeros((coord_features.size(0), self.stft_encoder[-2].out_features), device=self.device)

        parts = [stft_features]
        if depth_features is not None:
            parts.insert(0, depth_features)
        parts.append(coord_features)
        features = torch.cat(parts, dim=1)

        o = torch.relu(self.fc1(features))
        o = self.dropout(o)
        return self.fc2(o)

# ==========================================================
#  Helper Function: Random Sign-Sticky RIR Reconstruction
# ==========================================================

def reconstruct_random_sign_sticky(edc, stickiness=0.90):
    diff_edc = -np.diff(edc, append=0)
    diff_edc = np.clip(diff_edc, 0, None)
    rir_mag = np.sqrt(diff_edc)
    signs = np.empty_like(rir_mag)
    last_sign = 1
    for i, mag in enumerate(rir_mag):
        if mag == 0:
            signs[i] = last_sign
        else:
            signs[i] = last_sign if np.random.rand() < stickiness else -last_sign
            last_sign = signs[i]
    return rir_mag * signs

# ==========================================================
#  MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    # ------------------------------
    # Configuration
    # ------------------------------
    FS = 48000
    target_length = FS * 2   # 2 seconds
    rooms_to_process = 200
    absCases = 30
    max_files_to_load = rooms_to_process * absCases
    input_dim = 16

    room_features_csv = "dataset/room_acoustic_largedataset/roomFeaturesDataset.csv"
    edc_folder = "dataset/room_acoustic_largedataset/EDC"
    checkpoint_path = "Models/best_model.ckpt"
    scaler_X_path = f"Models/scaler_X_{max_files_to_load}_{target_length}.save"
    scaler_y_path = f"Models/scaler_edc_{max_files_to_load}_{target_length}.save"

    # Create output folder
    os.makedirs("inference_results", exist_ok=True)

    print("\n==============================")
    print("  RIR Reconstruction Inference")
    print("==============================")
    print("Select mode:")
    print("1 - Use existing dataset example")
    print("2 - Use custom room features")
    choice = input("Enter choice (1/2): ").strip()

    # ==========================================================
    #  OPTION 1: Existing Dataset
    # ==========================================================
    if choice == "1":
        print("\nYou selected: Use existing dataset")

        df_features = pd.read_csv(room_features_csv)
        room_ids = df_features.iloc[:, 0].values
        features_only = df_features.drop(columns=[df_features.columns[0]]).values

        rng = np.random.default_rng()
        rand_idx = rng.integers(0, 6000)
        selected_features = features_only[rand_idx]
        selected_room_id = room_ids[rand_idx]

        print(f"Selected room ID: {selected_room_id}")

        edc_filename = f"{selected_room_id}.npy"
        edc_path = os.path.join(edc_folder, edc_filename)
        if not os.path.exists(edc_path):
            raise FileNotFoundError(f"EDC file not found: {edc_path}")

        actual_edc = np.load(edc_path).flatten()
        actual_edc = np.pad(actual_edc, (0, max(0, target_length - len(actual_edc))), mode='constant')[:target_length]

        # Load scalers
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        # optional STFT scaler
        stft_scaler_path = 'Models/scaler_stft_acoustic_rooms.save'
        stft_stats = None
        if os.path.exists(stft_scaler_path):
            stft_stats = joblib.load(stft_scaler_path)

        # Scale input features
        new_features_scaled = scaler_X.transform(selected_features.reshape(1, -1))
        new_features_scaled = new_features_scaled.reshape(1, 1, input_dim)

        # Load model
        model = EDCModel.load_from_checkpoint(checkpoint_path, input_dim=input_dim, target_length=target_length, depth_enabled=False)
        model.eval()

        # Try to find the corresponding RIR wav file for the selected room id
        rir_path_candidate = None
        for root, dirs, files in os.walk('dataset'):
            for fn in files:
                if fn.endswith('.wav') and str(selected_room_id) in fn:
                    rir_path_candidate = os.path.join(root, fn)
                    break
            if rir_path_candidate:
                break

        stft_input = np.zeros((train_stft_freq_bins, train_stft_time_frames), dtype=np.float32)
        if rir_path_candidate and os.path.exists(rir_path_candidate):
            wav, sr = sf.read(rir_path_candidate)
            if wav.ndim > 1:
                wav = wav[:, 0]
            wav = np.asarray(wav, dtype=np.float32).flatten()
            wav = wav[:16000]
            if len(wav) < 16000:
                wav = np.pad(wav, (0, 16000 - len(wav)), mode='constant')
            stft_input = compute_stft_np(wav, sr)

        # normalize with saved stats if available
        if stft_stats is not None:
            stft_min = stft_stats['min']
            stft_max = stft_stats['max']
            stft_den = max(stft_max - stft_min, 1e-8)
            stft_input = (stft_input - stft_min) / stft_den

        with torch.no_grad():
            X_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)
            model._stft_tmp = torch.tensor(stft_input, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(X_tensor).cpu().numpy()
            model._stft_tmp = None

        pred_edc = scaler_y.inverse_transform(pred_scaled)[0]

        # MSE
        mse = mean_squared_error(actual_edc, pred_edc)
        print(f"EDC MSE = {mse:.6f}")

        # RIR reconstruction
        pred_rir = reconstruct_random_sign_sticky(pred_edc)
        actual_rir = reconstruct_random_sign_sticky(actual_edc)

        pred_rir = reconstruct_random_sign_sticky(pred_edc)
        actual_rir = reconstruct_random_sign_sticky(actual_edc)
        
        # Load the clap file
        clap, clap_sr = sf.read('clap22.wav')
        if clap.ndim > 1:
            clap = clap[:, 0]  # mono


        # Convolve
        pred_convolved  = fftconvolve(clap, pred_rir)
        actual_convolved = fftconvolve(clap, actual_rir)


        # Normalize to prevent clipping
        pred_convolved  = pred_convolved  / np.max(np.abs(pred_convolved)  + 1e-8)
        actual_convolved = actual_convolved / np.max(np.abs(actual_convolved) + 1e-8)

        # Save
        sf.write('pred_rir_convolved_original.wav',   pred_convolved.astype(np.float32),  clap_sr)
        sf.write('actual_rir_convolved_original.wav', actual_convolved.astype(np.float32), clap_sr)





        # Plot
        epsilon = 1e-18
        freqs = np.fft.rfftfreq(len(pred_rir), 1 / FS)
        pred_fft_db = 20 * np.log10(np.abs(np.fft.rfft(pred_rir)) / np.max(np.abs(np.fft.rfft(actual_rir))) + epsilon)
        actual_fft_db = 20 * np.log10(np.abs(np.fft.rfft(actual_rir)) / np.max(np.abs(np.fft.rfft(actual_rir))) + epsilon)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].plot(10 * np.log10(actual_edc + epsilon), label="Actual EDC")
        axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
        axs[0].set_title("EDCs (dB)"); axs[0].legend(); axs[0].grid(True)

        axs[1].plot(actual_rir, label="Actual RIR")
        axs[1].plot(pred_rir, label="Predicted RIR")
        axs[1].set_title("RIR"); axs[1].legend(); axs[1].grid(True)

        axs[2].plot(freqs, actual_fft_db, label="Actual FFT")
        axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
        axs[2].set_xscale("log"); axs[2].set_title("FFT"); axs[2].legend(); axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("inference_results/comparison_existing.png", dpi=300)
        plt.show()

    # ==========================================================
    #  OPTION 2: Custom Room Features
    # ==========================================================
    elif choice == "2":
        print("\nYou selected: Use custom room features")
        use_defaults = input("Use default example values? (y/n): ").strip().lower()

        if use_defaults == "y":
            length, width, height = 3.0, 4.0, 3.0
            src = [1.4, 1.4, 1.5]
            rec = [1.8, 3.0, 1.5]
            absorption = [0.14, 0.27, 0.36, 0.3, 0.24, 0.24, 0.03]
        else:
            length = float(input("Length (m): "))
            width = float(input("Width (m): "))
            height = float(input("Height (m): "))
            src = [float(input(f"Source {axis} (m): ")) for axis in ['X','Y','Z']]
            rec = [float(input(f"Receiver {axis} (m): ")) for axis in ['X','Y','Z']]
            absorption = [float(input(f"Absorption Band {i+1}: ")) for i in range(7)]

        selected_features = np.array([length, width, height] + src + rec + absorption)
        print(f"Custom feature vector: {selected_features}")

        # Predict only (no actual)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        new_features_scaled = scaler_X.transform(selected_features.reshape(1, -1)).reshape(1, 1, input_dim)
        model = EDCModel.load_from_checkpoint(checkpoint_path, input_dim=input_dim, target_length=target_length, depth_enabled=False)
        model.eval()

        # Ask for an optional WAV file to compute STFT for inference
        wav_path = input("Path to RIR/WAV file for STFT (leave empty to skip): ").strip()
        stft_input = np.zeros((train_stft_freq_bins, train_stft_time_frames), dtype=np.float32)
        if wav_path:
            if not os.path.exists(wav_path):
                print(f"Provided wav not found: {wav_path}")
            else:
                wav, sr = sf.read(wav_path)
                if wav.ndim > 1:
                    wav = wav[:, 0]
                wav = np.asarray(wav, dtype=np.float32).flatten()
                wav = wav[:16000]
                if len(wav) < 16000:
                    wav = np.pad(wav, (0, 16000 - len(wav)), mode='constant')
                stft_input = compute_stft_np(wav, sr)

        # normalize stft if scaler available
        stft_stats_path = 'Models/scaler_stft_acoustic_rooms.save'
        if os.path.exists(stft_stats_path):
            stft_stats = joblib.load(stft_stats_path)
            stft_min = stft_stats['min']
            stft_max = stft_stats['max']
            stft_input = (stft_input - stft_min) / max(stft_max - stft_min, 1e-8)

        with torch.no_grad():
            X_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)
            model._stft_tmp = torch.tensor(stft_input, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(X_tensor).cpu().numpy()
            model._stft_tmp = None
        pred_edc = scaler_y.inverse_transform(pred_scaled)[0]

        pred_rir = reconstruct_random_sign_sticky(pred_edc)
        freqs = np.fft.rfftfreq(len(pred_rir), 1 / FS)
        epsilon = 1e-18
        pred_fft_db = 20 * np.log10(np.abs(np.fft.rfft(pred_rir)) / np.max(np.abs(np.fft.rfft(pred_rir))) + epsilon)

        # Plot predicted only
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
        axs[0].set_title("Predicted EDC (dB)"); axs[0].legend(); axs[0].grid(True)

        axs[1].plot(pred_rir, label="Predicted RIR")
        axs[1].set_title("Predicted RIR"); axs[1].legend(); axs[1].grid(True)

        axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
        axs[2].set_xscale("log"); axs[2].set_title("Predicted FFT"); axs[2].legend(); axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("inference_results/predicted_only_custom.png", dpi=300)
        plt.show()

    else:
        print("❌ Invalid choice. Please run the script again.")
