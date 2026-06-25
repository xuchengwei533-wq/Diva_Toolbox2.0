import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import pyplot as plt


def list_wav_files(dataset_dir):
    wav_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.m4a', '.mp3', '.flac')):
                wav_files.append(os.path.join(root, file))
    wav_files.sort()
    return wav_files


def load_wav_file(file_path):
    try:
        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
    except Exception:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def preprocess_audio(audio, sr, target_sr=44100):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio, target_sr

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        threshold = peak * (10 ** (-30.0 / 20.0))
        active = np.flatnonzero(np.abs(audio) > threshold)
        if active.size > 0:
            pad = int(0.05 * target_sr)
            start = max(0, int(active[0]) - pad)
            end = min(audio.size, int(active[-1]) + pad)
            if end > start:
                audio = audio[start:end]
    return audio, target_sr


def vis_preprocess_comparison(original_audio, original_sr, processed_audio, processed_sr):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col', height_ratios=[2, 1, 2])
    fig.suptitle("Audio Preprocessing Comparison", fontsize=16, y=0.98)
    ax_wave_orig = axes[0, 0]
    ax_rms_orig = axes[1, 0]
    ax_spec_orig = axes[2, 0]
    ax_wave_proc = axes[0, 1]
    ax_rms_proc = axes[1, 1]
    ax_spec_proc = axes[2, 1]

    # 原始波形
    librosa.display.waveshow(original_audio, sr=original_sr, ax=ax_wave_orig, color='C0', alpha=0.8)
    ax_wave_orig.set_title("Original Waveform")
    ax_wave_orig.set_ylabel("Amplitude")
    ax_wave_orig.grid(True, alpha=0.3)
    # 原始 RMS 能量曲线
    rms_orig = librosa.feature.rms(y=original_audio)[0]
    times_orig = librosa.times_like(rms_orig, sr=original_sr)
    ax_rms_orig.plot(times_orig, rms_orig, color='C0', linewidth=1.5)
    ax_rms_orig.set_title("Original RMS Energy")
    ax_rms_orig.set_ylabel("RMS")
    ax_rms_orig.grid(True, alpha=0.3)
    # 原始 Spectrogram (dB)
    S_orig = np.abs(librosa.stft(original_audio))
    S_db_orig = librosa.amplitude_to_db(S_orig, ref=np.max)
    img_spec_orig = librosa.display.specshow(
        S_db_orig, sr=original_sr, x_axis='time', y_axis='log',
        ax=ax_spec_orig, cmap='magma'
    )
    ax_spec_orig.set_title("Original Spectrogram (dB)")
    fig.colorbar(img_spec_orig, ax=ax_spec_orig, format="%+2.0f dB", aspect=30)

    # 处理后波形（用相同 y 轴范围，便于对比幅度）
    y_max = max(np.abs(original_audio).max(), np.abs(processed_audio).max()) * 1.1
    librosa.display.waveshow(processed_audio, sr=processed_sr, ax=ax_wave_proc, color='C2', alpha=0.8)
    ax_wave_proc.set_title("Processed Waveform")
    ax_wave_proc.set_ylabel("Amplitude")
    ax_wave_proc.set_ylim(-y_max, y_max)  # 统一 y轴范围
    ax_wave_orig.set_ylim(-y_max, y_max)
    ax_wave_proc.grid(True, alpha=0.3)
    # 处理后 RMS 能量曲线
    rms_proc = librosa.feature.rms(y=processed_audio)[0]
    times_proc = librosa.times_like(rms_proc, sr=processed_sr)
    ax_rms_proc.plot(times_proc, rms_proc, color='C2', linewidth=1.5)
    ax_rms_proc.set_title("Processed RMS Energy")
    ax_rms_proc.set_ylabel("RMS")
    ax_rms_proc.grid(True, alpha=0.3)
    # 处理后 Spectrogram (dB)
    S_proc = np.abs(librosa.stft(processed_audio))
    S_db_proc = librosa.amplitude_to_db(S_proc, ref=np.max)
    img_spec_proc = librosa.display.specshow(
        S_db_proc, sr=processed_sr, x_axis='time', y_axis='log',
        ax=ax_spec_proc, cmap='magma'
    )
    ax_spec_proc.set_title("Processed Spectrogram (dB)")
    fig.colorbar(img_spec_proc, ax=ax_spec_proc, format="%+2.0f dB", aspect=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)


def load_audio(file_path, target_sr=44100, visualize=False):
    original_audio, original_sr = load_wav_file(file_path)
    processed_audio, processed_sr = preprocess_audio(original_audio, original_sr, target_sr)
    if visualize:
        vis_preprocess_comparison(original_audio, original_sr, processed_audio, processed_sr)
    return processed_audio, original_sr, target_sr


def load_score_matrix(path) -> pd.DataFrame:
    """
    从 Excel 文件加载评分矩阵，返回一个 DataFrame
    将第一列 wav 文件名设置为索引（audio_filename），后续列为各个声乐特征的评分
    """
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[!] Error loading score matrix from {path}: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    # 删去无数据的行
    mask_delete = df.apply(
        lambda row: row.astype(str).str.contains("删除", na=False).any(), axis=1
    )
    df = df[~mask_delete].copy()
    df = df.dropna(axis=0, how="all")
    # 将第一列设置为索引
    col_name = "audio_filename"
    if df.columns[0] != col_name:
        df = df.rename(columns={df.columns[0]: col_name})
    df = df.set_index(col_name)
    return df


def load_feat_series(csv_path):
    """从 CSV 文件加载特征序列，返回一个 1D 的 numpy 数组"""
    try:
        series = np.loadtxt(csv_path, delimiter=",")
    except Exception as e:
        print(f"[!] Error loading feature series from {csv_path}: {e}")
        return None
    if series is None or np.size(series) == 0:
        return None
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return None
    return series


if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    dataset_name = cfg.dataset.name
    acoustic_feats = cfg.acoustic_feats

    # 输入目录准备
    data_root = os.path.join(proj_root, "data")
    wav_dir = os.path.join(data_root, dataset_name)
    wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    wav_files = sorted(wav_files)

    for wav_file in wav_files:
        wav_path = os.path.join(wav_dir, wav_file)
        print(f"Processing: {wav_path}")
        audio, orig_sr, proc_sr = load_audio(wav_path, visualize=True)
