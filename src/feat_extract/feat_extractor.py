import os
import time
import json
from typing import Callable, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth as pm
from tqdm import tqdm

from src.data_loader import load_audio, load_feat_series


def compute_frame_level_f0_rms(audio, sr, hop_length=512, frame_length=2048):
    """计算整段 frame-level F0 与 RMS，并做长度对齐。"""
    if audio is None or len(audio) == 0 or sr <= 0:
        return None, None, None
    try:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
        )
    except Exception as e:
        print("[!] Error computing frame-level F0:", e)
        return None, None, None
    if f0 is None:
        return None, None, None

    f0 = np.asarray(f0, dtype=np.float32)
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0].astype(np.float32)
    min_len = min(f0.shape[0], rms.shape[0])
    if min_len <= 0:
        return None, None, None
    f0 = f0[:min_len]
    rms = rms[:min_len]
    times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)
    return f0, rms, times


def find_most_stable_window_by_f0_rms(
        audio,
        sr,
        hop_length=512,
        frame_length=2048,
        min_sec=1.0,
        max_sec=2.0,
        fallback_min_sec=0.3,
):
    """
    根据 F0/RMS 波动寻找最稳定窗口。
    优先寻找 1-2 秒的稳态段；若音频较短或 F0 不稳定，则逐步放宽条件，
    并使用连续发声段或高能量窗口作为保底，尽量保证每个非空音频都能返回可分析片段。
    返回 (audio_window, meta_info)。
    """
    f0, rms, _times = compute_frame_level_f0_rms(
        audio, sr, hop_length=hop_length, frame_length=frame_length
    )
    if f0 is None or rms is None:
        if audio is None or len(audio) == 0 or sr <= 0:
            return None, None
        # 如果 F0/RMS 本身无法可靠计算，则直接退化为整段中间片段。
        fallback_audio = np.asarray(audio, dtype=np.float32)
        return fallback_audio, {
            "start_time_sec": 0.0,
            "end_time_sec": float(len(fallback_audio) / sr),
            "duration_sec": float(len(fallback_audio) / sr),
            "window_frames": 0,
            "voiced_ratio": 0.0,
            "f0_std_cents": float("nan"),
            "rms_std_db": float("nan"),
            "selection_method": "full_audio_fallback",
        }

    n_frames = f0.shape[0]
    if n_frames <= 0:
        return None, None

    valid_mask = np.isfinite(f0) & (f0 > 0) & np.isfinite(rms) & (rms > 0)
    valid_count = int(np.sum(valid_mask))

    rms_db_all = np.full_like(rms, np.nan, dtype=np.float32)
    positive_rms = np.isfinite(rms) & (rms > 0)
    rms_db_all[positive_rms] = 20.0 * np.log10(rms[positive_rms] + 1e-12)

    f0_cents = np.full_like(f0, np.nan, dtype=np.float32)
    if valid_count >= 5:
        f0_valid = f0[valid_mask]
        rms_valid = rms_db_all[valid_mask]
        f0_med = float(np.median(f0_valid))
        if np.isfinite(f0_med) and f0_med > 0:
            f0_cents[valid_mask] = 1200.0 * np.log2(f0[valid_mask] / f0_med)
        f0_scale = float(np.nanstd(f0_cents[valid_mask]))
        rms_scale = float(np.nanstd(rms_valid))
    else:
        f0_scale = float("nan")
        rms_scale = float(np.nanstd(rms_db_all[positive_rms])) if np.any(positive_rms) else float("nan")

    if not np.isfinite(f0_scale) or f0_scale <= 1e-6:
        f0_scale = 1.0
    if not np.isfinite(rms_scale) or rms_scale <= 1e-6:
        rms_scale = 1.0

    min_frames_pref = max(5, int(np.ceil(min_sec * sr / hop_length)))
    max_frames_pref = max(min_frames_pref, int(np.floor(max_sec * sr / hop_length)))
    fallback_min_frames = max(5, int(np.ceil(fallback_min_sec * sr / hop_length)))
    max_frames = min(n_frames, max_frames_pref)
    if max_frames <= 0:
        max_frames = n_frames
    min_frames = min(max_frames, min_frames_pref)
    fallback_frames = min(max_frames, fallback_min_frames)
    if n_frames < min_frames:
        min_frames = max_frames
    if fallback_frames <= 0:
        fallback_frames = max_frames

    def build_meta(start_frame, end_frame, voiced_ratio, f0_std, rms_std, method):
        start_sample = int(start_frame * hop_length)
        end_sample = int(min(len(audio), end_frame * hop_length))
        if end_sample <= start_sample:
            end_sample = min(len(audio), start_sample + max(1, hop_length))
        if end_sample <= start_sample:
            return None, None
        window_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
        if window_audio.size == 0:
            return None, None
        meta = {
            "start_time_sec": float(start_sample / sr),
            "end_time_sec": float(end_sample / sr),
            "duration_sec": float((end_sample - start_sample) / sr),
            "window_frames": int(end_frame - start_frame),
            "voiced_ratio": float(voiced_ratio),
            "f0_std_cents": float(f0_std) if np.isfinite(f0_std) else float("nan"),
            "rms_std_db": float(rms_std) if np.isfinite(rms_std) else float("nan"),
            "selection_method": method,
        }
        return window_audio, meta

    def try_scored_search():
        best = None
        for voiced_threshold in [0.8, 0.6, 0.4, 0.2, 0.0]:
            for win_frames in range(max_frames, fallback_frames - 1, -1):
                for start in range(0, n_frames - win_frames + 1):
                    end = start + win_frames
                    win_valid = valid_mask[start:end]
                    voiced_ratio = float(np.mean(win_valid))
                    if voiced_ratio < voiced_threshold:
                        continue
                    seg_rms_db = rms_db_all[start:end]
                    seg_rms_db_valid = seg_rms_db[np.isfinite(seg_rms_db)]
                    if seg_rms_db_valid.size < max(3, int(np.ceil(max(voiced_threshold, 0.2) * win_frames))):
                        continue

                    seg_f0_cents = f0_cents[start:end][win_valid]
                    f0_std = float(np.std(seg_f0_cents)) if seg_f0_cents.size >= 3 else float("nan")
                    rms_std = float(np.std(seg_rms_db_valid))
                    f0_term = (f0_std / f0_scale) if np.isfinite(f0_std) else 2.0
                    rms_term = rms_std / rms_scale
                    duration_bonus = 0.05 * ((max_frames - win_frames) / max(max_frames, 1))
                    score = f0_term + rms_term + duration_bonus
                    candidate = (score, -voiced_ratio, -win_frames, start, end, f0_std, rms_std, voiced_threshold)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                break
        return best

    best = try_scored_search()
    if best is not None:
        _score, neg_voiced_ratio, neg_win_frames, start, end, f0_std, rms_std, voiced_threshold = best
        return build_meta(
            start,
            end,
            -neg_voiced_ratio,
            f0_std,
            rms_std,
            f"scored_search_thr_{voiced_threshold:.1f}",
        )

    if valid_count > 0:
        # 保底 1：使用最长连续发声区，并在需要时向两侧略做扩展。
        best_run = None
        run_start = None
        for idx, is_valid in enumerate(valid_mask):
            if is_valid and run_start is None:
                run_start = idx
            elif not is_valid and run_start is not None:
                candidate = (idx - run_start, run_start, idx)
                if best_run is None or candidate[0] > best_run[0]:
                    best_run = candidate
                run_start = None
        if run_start is not None:
            candidate = (n_frames - run_start, run_start, n_frames)
            if best_run is None or candidate[0] > best_run[0]:
                best_run = candidate

        if best_run is not None:
            run_len, run_start, run_end = best_run
            target_len = min(max_frames, max(run_len, fallback_frames))
            center = (run_start + run_end) // 2
            start = max(0, center - target_len // 2)
            end = min(n_frames, start + target_len)
            start = max(0, end - target_len)
            seg_valid = valid_mask[start:end]
            seg_f0_cents = f0_cents[start:end][seg_valid]
            seg_rms_db_valid = rms_db_all[start:end][np.isfinite(rms_db_all[start:end])]
            f0_std = float(np.std(seg_f0_cents)) if seg_f0_cents.size >= 3 else float("nan")
            rms_std = float(np.std(seg_rms_db_valid)) if seg_rms_db_valid.size >= 3 else float("nan")
            return build_meta(
                start,
                end,
                float(np.mean(seg_valid)) if seg_valid.size else 0.0,
                f0_std,
                rms_std,
                "longest_voiced_fallback",
            )

    # 保底 2：没有可靠 F0 时，直接选择高能量窗口。
    target_len = min(n_frames, max(fallback_frames, min(max_frames, int(np.ceil(min(1.0, len(audio) / float(sr)) * sr / hop_length)))))
    target_len = max(1, target_len)
    best_energy = None
    for start in range(0, n_frames - target_len + 1):
        end = start + target_len
        seg_rms = rms[start:end]
        finite_seg_rms = seg_rms[np.isfinite(seg_rms)]
        if finite_seg_rms.size == 0:
            continue
        mean_rms = float(np.mean(finite_seg_rms))
        candidate = (-mean_rms, start, end)
        if best_energy is None or candidate < best_energy:
            best_energy = candidate

    if best_energy is not None:
        _neg_rms, start, end = best_energy
        seg_valid = valid_mask[start:end]
        seg_rms_db_valid = rms_db_all[start:end][np.isfinite(rms_db_all[start:end])]
        rms_std = float(np.std(seg_rms_db_valid)) if seg_rms_db_valid.size >= 3 else float("nan")
        return build_meta(
            start,
            end,
            float(np.mean(seg_valid)) if seg_valid.size else 0.0,
            float("nan"),
            rms_std,
            "rms_fallback",
        )

    return build_meta(0, n_frames, 0.0, float("nan"), float("nan"), "full_signal_last_resort")


def save_stable_window_meta(output_dir, wav_filename, meta):
    """保存稳定段起止时间与稳定性指标。"""
    meta_dir = os.path.join(output_dir, "_stable_windows")
    os.makedirs(meta_dir, exist_ok=True)
    json_name = os.path.splitext(wav_filename)[0] + ".json"
    json_path = os.path.join(meta_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return json_path


def trim_edge_transients(audio, sr, head_ms=500, tail_ms=500, return_offsets=False):
    """
    去掉音频起始和结束瞬态：
    - 默认最多去掉前 500ms 和后 500ms。
    - 对短音频使用比例裁剪，避免一刀切导致无可分析内容。
    """
    if audio is None:
        empty = np.array([], dtype=np.float32)
        return (empty, 0, 0) if return_offsets else empty
    audio_arr = np.asarray(audio)
    if audio_arr.size == 0 or sr <= 0:
        empty = np.array([], dtype=np.float32)
        return (empty, 0, 0) if return_offsets else empty

    head_samples = int(sr * head_ms / 1000.0)
    tail_samples = int(sr * tail_ms / 1000.0)
    max_edge_trim = int(audio_arr.size * 0.15)
    head_samples = min(head_samples, max_edge_trim)
    tail_samples = min(tail_samples, max_edge_trim)
    min_keep_samples = max(1, int(sr * 0.3))
    total_trim = head_samples + tail_samples
    if total_trim <= 0:
        return (audio_arr, 0, 0) if return_offsets else audio_arr

    if audio_arr.size - total_trim < min_keep_samples:
        total_trim = max(0, audio_arr.size - min_keep_samples)
        head_samples = min(head_samples, total_trim // 2)
        tail_samples = min(tail_samples, total_trim - head_samples)

    if audio_arr.size <= head_samples + tail_samples:
        trimmed = audio_arr.astype(np.float32, copy=True)
        return (trimmed, 0, 0) if return_offsets else trimmed

    trimmed = audio_arr[head_samples: audio_arr.size - tail_samples]
    return (trimmed, head_samples, tail_samples) if return_offsets else trimmed


def extract_jitter(audio, sr, hop_length=512):
    # 优先使用 parselmouth（Praat 标准实现）
    try:
        snd = pm.Sound(audio, sampling_frequency=sr)
        point_process = pm.praat.call(
            snd, "To PointProcess (periodic, cc)",
            65.0, 1000.0
        )
        jitter_local = pm.praat.call(
            point_process, "Get jitter (local)",
            0.0, 0.0, 0.0001, 0.02, 1.3
        )
        if not np.isfinite(jitter_local):
            return None
        return np.asarray([float(jitter_local)], dtype=np.float32)
    except Exception as e:
        print("[!] Error extracting jitter with parselmouth:", e)
        return None


def extract_jitter_librosa(audio, sr, hop_length=512):
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    if f0 is None:
        return None
    f0 = np.asarray(f0, dtype=np.float32)
    mask = np.isfinite(f0) & (f0 > 0)
    f0_valid = f0[mask]
    if f0_valid.size < 2:
        return None
    periods = 1.0 / f0_valid
    diffs = np.abs(np.diff(periods))
    denom = np.mean(periods)
    if denom <= 0:
        return None
    return diffs / denom


def extract_shimmer(audio, sr, hop_length=512, frame_length=2048):
    """
    提取 Praat 周期级 Shimmer(local)。
    说明：保留 hop_length/frame_length 参数以兼容旧接口，但该实现不依赖它们。
    """
    _ = hop_length
    _ = frame_length
    try:
        snd = pm.Sound(audio, sampling_frequency=sr)
        point_process = pm.praat.call(
            snd, "To PointProcess (periodic, cc)",
            65.0, 1000.0
        )
        shimmer_local = pm.praat.call(
            [snd, point_process], "Get shimmer (local)",
            0.0, 0.0,        # from_time, to_time
            0.0001, 0.02,    # period_floor, period_ceiling
            1.3, 1.6         # maximum_period_factor, maximum_amplitude_factor
        )
        if not np.isfinite(shimmer_local):
            return None
        return np.asarray([float(shimmer_local)], dtype=np.float32)
    except Exception as e:
        print("[!] Error extracting shimmer(local) with parselmouth:", e)
        return None


def _local_peak_db(mag, freqs, target_hz, half_window_hz):
    f_min = max(0.0, target_hz - half_window_hz)
    f_max = target_hz + half_window_hz
    idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    if idx.size == 0:
        return None
    peak_mag = float(np.max(mag[idx]))
    if not np.isfinite(peak_mag) or peak_mag <= 0:
        return None
    return float(20.0 * np.log10(peak_mag + 1e-8))


def extract_h1h2(audio, sr, hop_length=512, n_fft=2048):
    _ = hop_length
    try:
        snd = pm.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(
            time_step=0.01,
            pitch_floor=65.0,
            pitch_ceiling=2093.0,
        )
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
        if f0_values.size == 0:
            return None
        f0_med = float(np.median(f0_values))
    except Exception as e:
        print("[!] Error estimating F0 for H1H2:", e)
        return None

    if not np.isfinite(f0_med) or f0_med <= 0 or 2.0 * f0_med >= 0.49 * sr:
        return None
    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.size < 16:
        return None
    audio_arr = audio_arr - float(np.mean(audio_arr))
    fft_size = int(2 ** np.ceil(np.log2(max(int(n_fft), audio_arr.size))))
    window = np.hanning(audio_arr.size).astype(np.float32)
    mag = np.abs(np.fft.rfft(audio_arr * window, n=fft_size))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sr))
    win1_hz = max(30.0, 0.1 * f0_med)
    win2_hz = max(30.0, 0.2 * f0_med)
    h1_db = _local_peak_db(mag, freqs, f0_med, win1_hz)
    h2_db = _local_peak_db(mag, freqs, 2.0 * f0_med, win2_hz)
    if h1_db is None or h2_db is None:
        return None
    return np.asarray([h1_db - h2_db], dtype=np.float32)


def extract_hnr(audio, sr, frame_length=2048, hop_length=512):
    snd = pm.Sound(audio, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity(time_step=hop_length / float(sr))
    hnr = harmonicity.values[0]
    return hnr


def extract_hnr_librosa(audio, sr, frame_length=2048, hop_length=512):
    harmonic = librosa.effects.harmonic(audio)
    noise = audio - harmonic
    rms_h = librosa.feature.rms(y=harmonic, frame_length=frame_length, hop_length=hop_length)[0]
    rms_n = librosa.feature.rms(y=noise, frame_length=frame_length, hop_length=hop_length)[0]
    hnr = 10.0 * np.log10((rms_h ** 2) / (rms_n ** 2 + 1e-12) + 1e-12)
    return hnr


def extract_q1(audio, sr, n_fft=2048, hop_length=512):
    """
    提取标准 Q1：Q1 = F1 / BW1
    - F1: 第一共振峰中心频率（Hz）
    - BW1: 第一共振峰带宽（Hz）
    使用 Praat/Burg formant 估计逐帧计算 Q1 序列。
    """
    _ = n_fft  # 兼容旧接口参数，Q1 计算不依赖 STFT FFT 点数
    try:
        snd = pm.Sound(audio, sampling_frequency=sr)
        time_step = hop_length / float(sr) if sr > 0 else 0.01
        max_formant = min(5500.0, 0.45 * sr)
        formant_obj = pm.praat.call(
            snd,
            "To Formant (burg)",
            time_step,      # time step
            5.0,            # max number of formants
            max_formant,    # max formant (Hz)
            0.025,          # window length (s)
            50.0,           # pre-emphasis from (Hz)
        )

        n_frames = int(pm.praat.call(formant_obj, "Get number of frames"))
        q1_vals = []
        for frame_idx in range(1, n_frames + 1):
            t = pm.praat.call(formant_obj, "Get time from frame number", frame_idx)
            f1 = pm.praat.call(formant_obj, "Get value at time", 1, t, "Hertz", "Linear")
            bw1 = pm.praat.call(formant_obj, "Get bandwidth at time", 1, t, "Hertz", "Linear")
            if not np.isfinite(f1) or not np.isfinite(bw1):
                continue
            if f1 <= 0 or bw1 <= 0:
                continue
            # 约束在合理语音范围内，减少跟踪异常点
            if f1 < 100.0 or f1 > 1500.0 or bw1 > 2000.0:
                continue
            q1_vals.append(float(f1 / bw1))

        if len(q1_vals) == 0:
            return None
        return np.asarray(q1_vals, dtype=np.float32)
    except Exception as e:
        print("[!] Error extracting Q1 with parselmouth:", e)
        return None


def _window_spectrum(audio, sr, n_fft=2048):
    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.size < 16 or sr <= 0:
        return None, None
    audio_arr = audio_arr - float(np.mean(audio_arr))
    fft_size = int(2 ** np.ceil(np.log2(max(int(n_fft), audio_arr.size))))
    window = np.hanning(audio_arr.size).astype(np.float32)
    mag = np.abs(np.fft.rfft(audio_arr * window, n=fft_size))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sr))
    return mag.astype(np.float64), freqs.astype(np.float64)


def extract_spectral_slope(audio, sr, hop_length=512, n_fft=2048):
    _ = hop_length
    mag, freqs = _window_spectrum(audio, sr, n_fft=n_fft)
    if mag is None:
        return None
    mask = (freqs >= 80.0) & (freqs <= min(8000.0, 0.45 * sr)) & np.isfinite(mag)
    if np.count_nonzero(mask) < 3:
        return None
    log_mag = np.log10(mag[mask] + 1e-8)
    slope = np.polyfit(freqs[mask], log_mag, 1)[0]
    if not np.isfinite(slope):
        return None
    return np.asarray([slope], dtype=np.float32)


def extract_low_freq_energy_ratio(audio, sr, hop_length=512, n_fft=2048):
    _ = hop_length
    mag, freqs = _window_spectrum(audio, sr, n_fft=n_fft)
    if mag is None:
        return None
    power = mag ** 2
    low_mask = (freqs >= 0) & (freqs <= 500)
    total_mask = (freqs >= 0) & (freqs <= 1000)
    total = float(np.sum(power[total_mask]))
    if total <= 0 or not np.isfinite(total):
        return None
    ratio = float(np.sum(power[low_mask]) / (total + 1e-12))
    return np.asarray([ratio], dtype=np.float32)


def extract_high_freq_noise_ratio(audio, sr, hop_length=512, n_fft=2048):
    _ = hop_length
    mag, freqs = _window_spectrum(audio, sr, n_fft=n_fft)
    if mag is None:
        return None
    power = mag ** 2
    high_min = min(4000.0, 0.45 * sr)
    high_mask = freqs >= high_min
    total = float(np.sum(power))
    if total <= 0 or not np.isfinite(total):
        return None
    ratio = float(np.sum(power[high_mask]) / (total + 1e-12))
    return np.asarray([ratio], dtype=np.float32)


def extract_cpp(audio, sr, hop_length=512, n_fft=2048):
    _ = hop_length
    mag, _freqs = _window_spectrum(audio, sr, n_fft=n_fft)
    if mag is None:
        return None
    log_mag = np.log(mag + 1e-8)
    cep = np.fft.irfft(log_mag)
    quef = np.arange(cep.shape[0]) / float(sr)
    qmin = 1.0 / 400.0
    qmax = 1.0 / 60.0
    mask = (quef >= qmin) & (quef <= qmax)
    if not np.any(mask):
        return None
    cep_range = cep[mask]
    cpp = float(np.max(cep_range) - np.mean(cep_range))
    if not np.isfinite(cpp):
        return None
    return np.asarray([cpp], dtype=np.float32)


def save_feat_series(out_dir, name, series):
    out_path = str(os.path.join(out_dir, name))
    np.savetxt(out_path, series, delimiter=",", fmt="%.6f")
    return out_path


def vis_feat_series(series, title, xlabel, ylabel, output_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 特征的时序图
    ax1.plot(series, linewidth=0.8)
    ax1.set_title(f"{title} - Time Series")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)

    # 特征的分布图
    ax2.hist(series, bins=50, alpha=0.7, edgecolor="black")
    ax2.set_title(f"{title} - Distribution")
    ax2.set_xlabel(ylabel)
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    # 绘制统计信息
    mean_val = np.mean(series)
    std_val = np.std(series)
    median_val = np.median(series)
    ax2.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
    ax2.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.4f}")
    ax2.legend()

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def extract_feats_from_single_wav(
        wav_filename,
        output_dir,
        audio,
        sr,
        visualize=False,
        overwrite=False
):
    """
    从单个 WAV 文件中提取声乐特征，并保存为 CSV 文件。

    Args:
        wav_filename: WAV 文件名（不包含路径），用于生成对应的 CSV 和 PNG 文件名
        output_dir: 原始特征输出目录，提取的特征 CSV 文件将保存在此目录下的对应子目录中
        audio: 音频数据数组
        sr: 采样率
        visualize: 是否可视化特征序列并保存为 PNG 文件
        overwrite: 是否覆盖已存在的特征 CSV 文件（默认为 False，即如果 CSV 已存在则跳过提取）
    """
    csv_filename = os.path.splitext(wav_filename)[0] + ".csv"
    png_filename = os.path.splitext(wav_filename)[0] + ".png"
    # 统一弱化起止瞬态，但对短音频采用自适应裁剪而不是固定硬切。
    head_ms = 500
    tail_ms = 500
    audio, head_trim_samples, tail_trim_samples = trim_edge_transients(
        audio, sr, head_ms=head_ms, tail_ms=tail_ms, return_offsets=True
    )
    if audio is None or len(audio) == 0:
        return [("StableWindow", "失败"), ("AllFeatures", "失败")]

    # 计算整段 frame-level F0/RMS，并优先选取 1-2 秒的稳态窗口；
    # 对短音频逐步放宽条件，确保尽量为每条音频返回一个可分析片段。
    if len(audio) > int(6.0 * sr):
        target_samples = max(1, int(2.0 * sr))
        stable_start_samples = max(0, len(audio) // 2 - target_samples // 2)
        stable_end_samples = min(len(audio), stable_start_samples + target_samples)
        stable_audio = np.asarray(audio[stable_start_samples:stable_end_samples], dtype=np.float32)
        stable_meta = {
            "start_time_sec": float(stable_start_samples / sr),
            "end_time_sec": float(stable_end_samples / sr),
            "duration_sec": float((stable_end_samples - stable_start_samples) / sr),
            "window_frames": 0,
            "voiced_ratio": float("nan"),
            "f0_std_cents": float("nan"),
            "rms_std_db": float("nan"),
            "selection_method": "center_2s_long_audio",
        }
    else:
        stable_audio, stable_meta = find_most_stable_window_by_f0_rms(
            audio,
            sr,
            hop_length=512,
            frame_length=2048,
            min_sec=1.0,
            max_sec=2.0,
            fallback_min_sec=0.3,
        )
    if stable_audio is None or stable_meta is None or len(stable_audio) == 0:
        print(f"[!] {wav_filename} 稳定段提取失败，跳过该文件。")
        return [("StableWindow", "失败"), ("AllFeatures", "失败")]

    # 记录稳定段在原始输入音频中的时间，并保存自适应裁剪后的实际偏移量。

    stable_meta["start_time_sec_after_trim"] = stable_meta["start_time_sec"]
    stable_meta["end_time_sec_after_trim"] = stable_meta["end_time_sec"]
    stable_meta["trim_head_sec"] = float(head_trim_samples / sr)
    stable_meta["trim_tail_sec"] = float(tail_trim_samples / sr)
    stable_meta["start_time_sec_in_original"] = stable_meta["start_time_sec"] + float(head_trim_samples / sr)
    stable_meta["end_time_sec_in_original"] = stable_meta["end_time_sec"] + float(head_trim_samples / sr)
    save_stable_window_meta(output_dir, wav_filename, stable_meta)
    audio = stable_audio

    targets = [
        ("Jitter", extract_jitter, (audio, sr)),
        ("Shimmer", extract_shimmer, (audio, sr)),
        ("H1H2_output", extract_h1h2, (audio, sr)),
        ("HNR", extract_hnr, (audio, sr)),
        ("Q1", extract_q1, (audio, sr)),
        ("SpectralSlope", extract_spectral_slope, (audio, sr)),
        ("LowFreqEnergyRatio", extract_low_freq_energy_ratio, (audio, sr)),
        ("HighFreqNoiseRatio", extract_high_freq_noise_ratio, (audio, sr)),
        ("CPP", extract_cpp, (audio, sr)),
    ]

    task_log = []
    for feat_name, extract_func, args in targets:
        out_dir = str(os.path.join(output_dir, feat_name))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, csv_filename)
        if not overwrite and os.path.exists(out_path):
            task_log.append((feat_name, "跳过"))
            continue
        feat_series = extract_func(*args)
        if feat_series is None:
            task_log.append((feat_name, "失败"))
            continue
        save_feat_series(out_dir, csv_filename, feat_series)
        task_log.append((feat_name, "完成"))
        if visualize:
            vis_path = os.path.join(out_dir, png_filename)
            vis_feat_series(
                feat_series,
                title=f"{feat_name} - {wav_filename}",
                xlabel="Frame",
                ylabel=feat_name,
                output_path=vis_path
            )
    return task_log


def extract_feats_from_wav_dir(
        wav_dir,
        output_dir,
        visualize=False,
        overwrite=False,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
):
    """
    从指定目录下的所有 WAV 文件中提取声乐特征，并保存为 CSV 文件。
    每个 WAV 文件的每个声学特征对应一个 CSV 文件，保存在 output_dir 下的对应特征子目录中。
    """
    logger = log_callback if log_callback is not None else print
    # 输入准备
    if not os.path.isdir(wav_dir):
        logger(f"[!] Directory not found: {wav_dir}")
        return
    wav_fullpaths = []
    for root, _dirs, files in os.walk(wav_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_fullpaths.append(os.path.join(root, f))
    wav_fullpaths = sorted(wav_fullpaths)
    logger(f"[*] 在目录 {wav_dir}（含子目录）中发现 {len(wav_fullpaths)} 个 WAV 文件，准备提取特征...")
    if len(wav_fullpaths) == 0:
        logger("[!] 未发现 WAV 文件，提取结束。")
        return

    # 输出准备
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger(f"[*] 提取的特征将保存在目录 {output_dir} 下的对应子目录中。")
    if progress_callback is not None:
        progress_callback(0, len(wav_fullpaths), "", "准备开始")

    # 遍历 WAV 文件，提取特征
    pbar = tqdm(
        wav_fullpaths, total=len(wav_fullpaths),
        desc="提取特征", unit="文件", dynamic_ncols=True
    )
    for idx, wav_fullpath in enumerate(pbar, start=1):
        start_t = time.perf_counter()
        # 使用相对路径避免不同子目录同名文件覆盖
        rel_path = os.path.relpath(wav_fullpath, wav_dir)
        wav_file = rel_path.replace(os.sep, "__")
        # 加载单个音频文件
        audio, original_sr, target_sr = load_audio(wav_fullpath)
        # 提取特征并保存为 CSV 文件
        results = extract_feats_from_single_wav(
            wav_file, output_dir, audio, target_sr,
            visualize=visualize,  # 原始特征序列可视化
            overwrite=overwrite,  # 已存在的特征 CSV 文件是否被覆盖（重新提取）
        )
        cost_s = time.perf_counter() - start_t
        # 更新进度条和日志
        status = ",".join([f"{k}:{v}" for k, v in results])
        pbar.set_postfix({"步骤": f"{cost_s:.1f}s", "文件": rel_path})
        if log_callback is None:
            tqdm.write(f"[{idx}/{len(wav_fullpaths)}] {rel_path} | {status}")
        logger(f"[{idx}/{len(wav_fullpaths)}] {rel_path} | {status}")
        if progress_callback is not None:
            progress_callback(idx, len(wav_fullpaths), rel_path, status)
    logger("[+] 所有文件的特征提取已完成！")


def extract_feats_stats_from_csv(raw_feats_dir, output_dir=None) -> pd.DataFrame:
    """
    提取所有音频文件的各个声学特征的各项统计信息，生成汇总字典。
    具体而言，从 CSV 文件中提取各个声学特征序列（feats series）的统计量（feats stats）。
    第一列为音频文件名 audio_name，后续列为各个声学特征的统计信息，如 HNR。
    若 output_dir 不为 None，则将提取的统计信息保存为 CSV 文件。
    """
    if not os.path.isdir(raw_feats_dir):
        print(f"[!] Directory not found: {raw_feats_dir}")
        return pd.DataFrame()
    # 获取所有被提取的特征名称（即 raw_feats_dir 下的子目录名）
    feat_names = [
        d for d in os.listdir(raw_feats_dir)
        if os.path.isdir(os.path.join(raw_feats_dir, d))
    ]

    # 提取为字典: { "audio_filename": { "HNR": 12.5, "CPP": 8.2, ... }, ... }
    stats = {}
    for feat_name in sorted(feat_names):
        feat_dir = os.path.join(raw_feats_dir, feat_name)
        csv_files = [f for f in os.listdir(feat_dir) if f.lower().endswith(".csv")]
        for csv_file in csv_files:
            wav_filename = os.path.splitext(csv_file)[0] + ".wav"
            csv_path = os.path.join(feat_dir, csv_file)
            try:
                series = load_feat_series(csv_path)
                if series.size == 0:
                    continue
                if wav_filename not in stats:
                    stats[wav_filename] = {}
                # 提取声学特征序列的中位数作为统计信息（也可添加/改为均值或其他统计量）
                stats[wav_filename][feat_name] = np.median(series)
            except Exception as e:
                print(f"[!] Error processing {csv_path}: {e}")

    # 转换为 DataFrame，行索引为音频文件名，列为各个声学特征的统计信息
    df_stats = pd.DataFrame.from_dict(stats, orient="index")
    df_stats.index.name = "audio_filename"
    # 可选地保存为 CSV 文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "feats_data.csv")
        df_stats.to_csv(output_csv_path)
        print(f"[+] Saved feature statistics summary to {output_csv_path}")
    return df_stats


if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    dataset_name = cfg.dataset.name
    score_file = cfg.dataset.score_file
    acoustic_feats = cfg.acoustic_feats

    # 输入输出目录准备
    data_root = os.path.join(proj_root, "data")
    wav_dir = os.path.join(data_root, dataset_name)
    outputs_root = os.path.join(proj_root, "outputs")
    raw_feats_dir = os.path.join(outputs_root, "raw_feats", dataset_name)
    for feat in acoustic_feats:
        os.makedirs(os.path.join(raw_feats_dir, feat), exist_ok=True)

    print("[*] 开始从 WAV 文件中提取声学特征...")
    extract_feats_from_wav_dir(wav_dir, raw_feats_dir, visualize=True, overwrite=False)

    print("[*] 开始提取特征统计信息...")
    df_stats = extract_feats_stats_from_csv(raw_feats_dir, outputs_root)
    print(f"[+] 提取完成，共 {len(df_stats)} 个音频文件的特征统计信息已保存。")

    print(f"[*] 加载评分矩阵：{score_file}")
    from src.data_loader import load_score_matrix
    score_path = os.path.join(data_root, dataset_name, score_file)
    df_score = load_score_matrix(score_path)

    print("[*] 开始合并评分矩阵和特征统计信息...")
    from src.combined_data import CombinedData
    combined_data = CombinedData(df_score, df_stats)
    combined_data.save_to_csv(output_dir=outputs_root)
    print("[+] 合并完成！")

