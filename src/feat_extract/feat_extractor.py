import os
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth as pm
from tqdm import tqdm

from src.data_loader import load_audio, load_feat_series


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
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    if rms is None or rms.size == 0:
        return None
    min_len = min(f0.shape[0], rms.shape[0])
    f0 = f0[:min_len]
    rms = rms[:min_len]
    mask = np.isfinite(f0) & (f0 > 0)
    amp = rms[mask]
    if amp.size < 2:
        return None
    diffs = np.abs(np.diff(amp))
    denom = np.mean(amp)
    if denom <= 0:
        return None
    return diffs / denom


def extract_h1h2(audio, sr, hop_length=512, n_fft=2048):
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
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    n_bins, n_frames = S.shape
    min_len = min(f0.shape[0], n_frames)
    f0 = f0[:min_len]
    S = S[:, :min_len]
    h1h2_vals = []
    for t in range(min_len):
        f0_t = f0[t]
        if not np.isfinite(f0_t) or f0_t <= 0:
            continue
        bin1 = int(np.round(f0_t * n_fft / sr))
        bin2 = int(np.round(2.0 * f0_t * n_fft / sr))
        if bin2 <= 0 or bin2 >= n_bins or bin1 <= 0 or bin1 >= n_bins:
            continue
        h1 = S[bin1, t]
        h2 = S[bin2, t]
        h1_db = 20.0 * np.log10(h1 + 1e-8)
        h2_db = 20.0 * np.log10(h2 + 1e-8)
        h1h2_vals.append(h1_db - h2_db)
    if len(h1h2_vals) == 0:
        return None
    return np.asarray(h1h2_vals, dtype=np.float32)


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


def extract_q_values(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    q_vals = []
    for t in range(S.shape[1]):
        mag = S[:, t]
        if mag.size == 0:
            continue
        peak_idx = int(np.argmax(mag))
        peak_mag = mag[peak_idx]
        if not np.isfinite(peak_mag) or peak_mag <= 0:
            continue
        f0 = freqs[peak_idx]
        if not np.isfinite(f0) or f0 <= 0:
            continue
        target = peak_mag / np.sqrt(2.0)
        left = peak_idx
        while left > 0 and mag[left] >= target:
            left -= 1
        right = peak_idx
        while right < len(mag) - 1 and mag[right] >= target:
            right += 1
        if left == peak_idx or right == peak_idx:
            continue
        bw = freqs[right] - freqs[left]
        if not np.isfinite(bw) or bw <= 0:
            continue
        q_vals.append(float(f0 / bw))
    if len(q_vals) == 0:
        return None
    return np.asarray(q_vals, dtype=np.float32)


def extract_spectral_slope(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    if S is None or S.size == 0:
        return None
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    slopes = []
    for t in range(S.shape[1]):
        mag = S[:, t]
        log_mag = np.log10(mag + 1e-8)
        slope = np.polyfit(freqs, log_mag, 1)[0]
        slopes.append(slope)
    if len(slopes) == 0:
        return None
    return np.asarray(slopes, dtype=np.float32)


def extract_low_freq_energy_ratio(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 0) & (freqs <= 500)
    total_mask = (freqs >= 0) & (freqs <= 1000)
    low_energy = np.sum(S[low_mask, :], axis=0)
    total_energy = np.sum(S[total_mask, :], axis=0)
    ratio = low_energy / (total_energy + 1e-12)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return None
    return ratio.astype(np.float32)


def extract_high_freq_noise_ratio(audio, sr, hop_length=512, n_fft=2048):
    harmonic = librosa.effects.harmonic(audio)
    noise = audio - harmonic
    S = np.abs(librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_min = min(4000.0, 0.45 * sr)
    high_mask = freqs >= high_min
    high_energy = np.sum(S[high_mask, :], axis=0)
    total_energy = np.sum(S, axis=0)
    ratio = high_energy / (total_energy + 1e-12)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return None
    return ratio.astype(np.float32)


def extract_cpp(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    if S is None or S.size == 0:
        return None
    log_mag = np.log(S + 1e-8)
    cepstra = np.fft.irfft(log_mag, axis=0)
    quef = np.arange(cepstra.shape[0]) / float(sr)
    qmin = 1.0 / 400.0
    qmax = 1.0 / 60.0
    mask = (quef >= qmin) & (quef <= qmax)
    if not np.any(mask):
        return None
    cep_range = cepstra[mask, :]
    peak = np.max(cep_range, axis=0)
    baseline = np.mean(cep_range, axis=0)
    cpp = peak - baseline
    cpp = cpp[np.isfinite(cpp)]
    if cpp.size == 0:
        return None
    return cpp.astype(np.float32)


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

    targets = [
        ("Jitter", extract_jitter, (audio, sr)),
        ("Shimmer", extract_shimmer, (audio, sr)),
        ("H1H2", extract_h1h2, (audio, sr)),
        ("HNR", extract_hnr, (audio, sr)),
        ("QValue", extract_q_values, (audio, sr)),
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
        overwrite=False
):
    """
    从指定目录下的所有 WAV 文件中提取声乐特征，并保存为 CSV 文件。
    每个 WAV 文件的每个声学特征对应一个 CSV 文件，保存在 output_dir 下的对应特征子目录中。
    """
    # 输入准备
    if not os.path.isdir(wav_dir):
        print(f"[!] Directory not found: {wav_dir}")
        return
    wav_fullpaths = []
    for root, _dirs, files in os.walk(wav_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_fullpaths.append(os.path.join(root, f))
    wav_fullpaths = sorted(wav_fullpaths)
    print(f"[*] 在目录 {wav_dir}（含子目录）中发现 {len(wav_fullpaths)} 个 WAV 文件，准备提取特征...")
    if len(wav_fullpaths) == 0:
        print("[!] 未发现 WAV 文件，提取结束。")
        return

    # 输出准备
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 提取的特征将保存在目录 {output_dir} 下的对应子目录中。")

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
        tqdm.write(f"[{idx}/{len(wav_fullpaths)}] {rel_path} | {status}")
    print("[+] 所有文件的特征提取已完成！")


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

