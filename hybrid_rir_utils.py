"""Hybrid RIR: predicted early segment + ground-truth tail with RMS matched at the splice."""

import numpy as np
import pandas as pd
from typing import Optional
from rir_metrics import estimate_t60_t20

# Match RMS using the last / first N samples of early / tail (cf. batch RMS over [-160:] / [:160]).
JUNCTION_WIN = 160

# Must match ``inference_edcModelPytorchLighteningV3.N_REF_RIRS`` / ref-RIR encoder input.
REF_RIR_COUNT = 3


def ref_rir_rows_by_room(df: pd.DataFrame, n_refs: int = REF_RIR_COUNT) -> dict:
    """First ``n_refs`` manifest rows per room — same selection as ``build_ref_rir_feature_lookup``."""
    ref_by_room = {}
    for (room_type, room_id), group in df.groupby(["room_type", "room_id"]):
        ref_by_room[(room_type, room_id)] = group.head(n_refs).reset_index(drop=True)
    return ref_by_room


def hybrid_pred_gt_tail_wrong_t60(pred, tgt, n_head,
                                   t60_scale=1.1, sample_rate=16000):
    """
    Use GT tail but with rescaled decay rate.
    t60_scale > 1 makes tail decay slower (too reverberant)
    t60_scale < 1 makes tail decay faster (too dry)
    """
    p = np.asarray(pred, dtype=np.float64).flatten()
    t = np.asarray(tgt, dtype=np.float64).flatten()
    L = min(len(p), len(t))
    p, t = p[:L], t[:L]
    
    pred_early = p[:n_head]
    tgt_late = t[n_head:].copy()
    
    # Estimate actual T60 of tail
    t60_actual = estimate_t60_t20(tgt_late, sample_rate)
    if not np.isfinite(t60_actual):
        t60_actual = 0.5
    
    # Apply correction envelope to change effective T60
    # multiply by exp(correction) where correction adjusts decay rate
    time = np.arange(len(tgt_late)) / sample_rate
    t60_target = t60_actual * t60_scale
    # correction = ratio of desired decay to actual decay
    correction = np.exp(-6.9078 * time * (1/t60_target - 1/t60_actual))
    modified_tail = tgt_late * correction
    
    # RMS match at junction
    j = min(160, len(pred_early), len(modified_tail))
    early_end_rms = np.sqrt(np.mean(pred_early[-j:]**2) + 1e-20)
    tail_start_rms = np.sqrt(np.mean(modified_tail[:j]**2) + 1e-20) + 1e-8
    scale = early_end_rms / tail_start_rms
    
    return np.concatenate([pred_early, 
                          modified_tail * scale]).astype(np.float32)

def hybrid_pred_gt_tail_scaled(
    pred: np.ndarray,
    tgt: np.ndarray,
    n_head: int,
    junction_win: Optional[int] = None,
) -> np.ndarray:
    """
    Take early samples from ``pred`` and the remainder from ``tgt``, scaling the tail so
    RMS(pred_early[-j:]) matches RMS(tgt_late[:j]) with j = min(junction_win, …).

    Removes amplitude discontinuity at the junction for fair hybrid metrics.
    """
    if junction_win is None:
        junction_win = JUNCTION_WIN
    p = np.asarray(pred, dtype=np.float64).flatten()
    t = np.asarray(tgt, dtype=np.float64).flatten()
    L = min(len(p), len(t))
    if L == 0:
        return np.array([], dtype=np.float32)
    p, t = p[:L], t[:L]
    n_head = min(int(n_head), L)
    pred_early = p[:n_head]
    tgt_late = t[n_head:]
    if len(tgt_late) == 0:
        return pred_early.astype(np.float32)
    j = min(int(junction_win), len(pred_early), len(tgt_late))
    if j < 1:
        scale = 1.0
    else:
        early_end = pred_early[-j:]
        tail_start = tgt_late[:j]
        early_end_rms = np.sqrt(np.mean(early_end**2) + 1e-20)
        tail_start_rms = np.sqrt(np.mean(tail_start**2) + 1e-20) + 1e-8
        scale = early_end_rms / tail_start_rms
    tgt_late_scaled = tgt_late * scale
    return np.concatenate([pred_early, tgt_late_scaled]).astype(np.float32)


DEFAULT_HYBRID_TAIL_NOISE_DB = -40.0


def hybrid_pred_gt_tail_noisy(
    pred: np.ndarray,
    tgt: np.ndarray,
    n_head: int,
    noise_db: float = DEFAULT_HYBRID_TAIL_NOISE_DB,
    junction_win: Optional[int] = None,
) -> np.ndarray:
    """
    Predicted early segment + ground-truth tail with small additive Gaussian noise on the tail,
    then RMS-match at the splice (same junction window as ``hybrid_pred_gt_tail_scaled``).

    ``noise_db`` sets the noise RMS relative to the tail RMS before noise (e.g. -40 =>
    noise RMS is 40 dB below tail RMS).
    """
    if junction_win is None:
        junction_win = JUNCTION_WIN
    p = np.asarray(pred, dtype=np.float64).flatten()
    t = np.asarray(tgt, dtype=np.float64).flatten()
    L = min(len(p), len(t))
    if L == 0:
        return np.array([], dtype=np.float32)
    p, t = p[:L], t[:L]
    n_head = min(int(n_head), L)
    pred_early = p[:n_head].copy()
    tgt_late = t[n_head:].copy()
    if len(tgt_late) == 0:
        return pred_early.astype(np.float32)

    tail_rms = np.sqrt(np.mean(tgt_late**2)) + 1e-10
    noise_amp = tail_rms * (10.0 ** (noise_db / 20.0))
    noise = np.random.randn(len(tgt_late)) * noise_amp
    noisy_tail = tgt_late + noise

    j = min(int(junction_win), len(pred_early), len(noisy_tail))
    if j < 1:
        scale = 1.0
    else:
        early_end = pred_early[-j:]
        tail_start = noisy_tail[:j]
        early_end_rms = np.sqrt(np.mean(early_end**2) + 1e-8)
        tail_start_rms = np.sqrt(np.mean(tail_start**2) + 1e-8)
        scale = early_end_rms / tail_start_rms
    noisy_tail_scaled = noisy_tail * scale
    return np.concatenate([pred_early, noisy_tail_scaled]).astype(np.float32)
