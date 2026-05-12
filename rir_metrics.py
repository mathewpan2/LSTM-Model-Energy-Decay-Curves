"""Shared RIR acoustic metrics (T20-based RT60 and percentage T60 error)."""

import numpy as np


def estimate_t60_t20(rir, sample_rate=16000):
    """
    RT60 from T20: Schroeder backward integral, linear decay between -5 dB and -25 dB;
    RT60 = 3 × T20 (ISO 3382-style).
    """
    rir = np.asarray(rir, dtype=np.float64).flatten()
    peak = int(np.abs(rir).argmax())
    tail = rir[peak:]
    if len(tail) < 256:
        return np.nan
    h2 = tail.astype(np.float64) ** 2
    sch = np.cumsum(h2[::-1])[::-1]
    if sch[0] < 1e-30:
        return np.nan
    sch_norm = sch / sch[0]
    db = 10.0 * np.log10(np.maximum(sch_norm, 1e-15))
    t = np.arange(len(db), dtype=np.float64) / float(sample_rate)
    idx5 = np.where(db <= -5.0)[0]
    idx25 = np.where(db <= -25.0)[0]
    if len(idx5) == 0 or len(idx25) == 0:
        return np.nan
    i0, i1 = int(idx5[0]), int(idx25[0])
    if i1 <= i0:
        return np.nan
    t_seg = t[i0 : i1 + 1]
    db_seg = db[i0 : i1 + 1]
    if len(t_seg) < 4:
        return np.nan
    slope, _ = np.polyfit(t_seg, db_seg, 1)
    if slope >= -1e-12:
        return np.nan
    t20 = 20.0 / abs(slope)
    rt60 = 3.0 * t20
    return float(np.clip(rt60, 0.05, 10.0))


def t60_percentage_error(pred, target, sample_rate=16000):
    t60_p = estimate_t60_t20(pred, sample_rate)
    t60_t = estimate_t60_t20(target, sample_rate)
    if not np.isfinite(t60_p) or not np.isfinite(t60_t) or t60_t < 1e-6:
        return np.nan
    return 100.0 * abs(t60_p - t60_t) / t60_t

def compute_drr(rir, sample_rate=16000, direct_ms=5):
    direct_samples = int(direct_ms * sample_rate / 1000)
    peak = int(np.abs(rir).argmax())
    direct_energy = np.sum(rir[peak:peak + direct_samples]**2) + 1e-10
    reverb_energy = np.sum(rir[peak + direct_samples:]**2) + 1e-10
    return 10 * np.log10(direct_energy / reverb_energy)

def estimate_edt(rir, sample_rate=16000):
    """Early decay time (s): Schroeder backward integral; EDT ≈ 6.908 × T10 (-10 dB on decay)."""
    rir = np.asarray(rir, dtype=np.float64).flatten()
    peak = int(np.abs(rir).argmax())
    tail = rir[peak:]
    if len(tail) < 64:
        return np.nan
    h2 = tail ** 2
    sch = np.cumsum(h2[::-1])[::-1]
    if sch[0] < 1e-30:
        return np.nan
    sch_norm = sch / sch[0]
    db = 10.0 * np.log10(np.maximum(sch_norm, 1e-15))
    t = np.arange(len(db)) / float(sample_rate)
    crossing = np.where(db <= -10.0)[0]
    if len(crossing) == 0:
        return np.nan
    i = int(crossing[0])
    if i == 0:
        t10 = float(t[0])
    else:
        t10 = float(np.interp(-10.0, db[i - 1 : i + 1], t[i - 1 : i + 1]))
    if not np.isfinite(t10):
        return np.nan
    edt = 6.908 * t10
    return float(np.clip(edt, 0.02, 10.0))

def estimate_c50(rir, sample_rate=16000, split_ms=50.0):
    """Clarity C50 (dB): early (≤50 ms after onset) vs late energy."""
    rir = np.asarray(rir, dtype=np.float64).flatten()
    peak = int(np.abs(rir).argmax())
    split = peak + int(split_ms * sample_rate / 1000.0)
    split = min(max(split, peak + 1), len(rir))
    early = np.sum(rir[peak:split] ** 2)
    late = np.sum(rir[split:] ** 2) + 1e-20
    return float(10.0 * np.log10(early / late + 1e-20))


def metrics_drr_edt_c50_t60(pred_wav, actual_wav, sample_rate=16000):
    """
    Absolute DRR / EDT / C50 errors and T60 percentage error between two waveforms
    (same definitions as ``train_single_room`` ablation / §8).
    """
    pw = np.asarray(pred_wav, dtype=np.float64).flatten()
    aw = np.asarray(actual_wav, dtype=np.float64).flatten()
    drr_e = abs(compute_drr(pw, sample_rate) - compute_drr(aw, sample_rate))
    edt_p, edt_a = estimate_edt(pw, sample_rate), estimate_edt(aw, sample_rate)
    edt_e = abs(edt_p - edt_a) if np.isfinite(edt_p) and np.isfinite(edt_a) else np.nan
    c50_p, c50_a = estimate_c50(pw, sample_rate), estimate_c50(aw, sample_rate)
    c50_e = abs(c50_p - c50_a) if np.isfinite(c50_p) and np.isfinite(c50_a) else np.nan
    t60_e = t60_percentage_error(pw, aw, sample_rate)
    return drr_e, edt_e, c50_e, t60_e


def waveform_metric_stat_box(ax, pred_wav, actual_wav, sample_rate=16000):
    """Overlay DRR / EDT / T60 / C50 in the upper-right (matches train_single_room ablation plots)."""
    drr_e, edt_e, c50_e, t60_e = metrics_drr_edt_c50_t60(pred_wav, actual_wav, sample_rate)
    lines = [
        f"DRR MAE: {drr_e:.2f} dB" if np.isfinite(drr_e) else "DRR MAE: n/a",
        f"EDT MAE: {edt_e:.3f} s" if np.isfinite(edt_e) else "EDT MAE: n/a",
        f"T60 err: {t60_e:.1f}%" if np.isfinite(t60_e) else "T60 err: n/a",
        f"C50 MAE: {c50_e:.2f} dB" if np.isfinite(c50_e) else "C50 MAE: n/a",
    ]
    ax.text(
        0.98,
        0.98,
        chr(10).join(lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88),
    )