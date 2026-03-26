import os
import numpy as np
import librosa
import parselmouth
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import variation
import matplotlib.pyplot as plt


# --- 辅助函数：去趋势 (Detrending) ---
def remove_trend(signal, window_size=0.2, sr_time_step=None):
    """
    移除信号中的低频趋势（旋律线），只保留高频波动（颤音）。
    使用移动平均作为趋势估计。
    """
    if len(signal) == 0:
        return signal

    if sr_time_step is None:
        # 如果不知道采样率时间步长，假设窗口大小为点数 (默认取信号长度的10%-20%)
        # 确保窗口至少为3，且不超过信号长度
        kernel_size = max(3, int(len(signal) * 0.15))
    else:
        kernel_size = int(window_size / sr_time_step)

    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= len(signal):
        return signal - np.mean(signal)

    # mode='same' 保证输出长度与输入一致
    trend = np.convolve(signal, np.ones(kernel_size) / kernel_size, mode='same')
    return signal - trend


# --- 辅助函数：正弦波模型 ---
def sine_model(t, a, f, p, c):
    """a: 幅度, f: 频率, p: 相位, c: 中心偏移"""
    return a * np.sin(2 * np.pi * f * t + p) + c


# ==============================================================================
# 🚀 核心修改：带严格数据清理的 F0 与包络提取
# ==============================================================================

def extract_f0_and_envelope_librosa(
        audio, sr, fmin=75.0, fmax=900.0,
        rms_threshold_ratio=0.05,
        prob_threshold=0.65,
        min_voiced_frames=50,
):
    """
    使用 librosa.pyin 提取 F0 和 RMS 能量包络，并执行严格的数据清理。

    清理策略:
    1. RMS 能量过滤：去除音量低于阈值一定比例的静音/噪声帧。
    2. 置信度过滤：去除 pyin 判定为"非有声"或置信度低的帧。
    3. 频率范围过滤：强制限制在 [fmin, fmax] 范围内，去除八度错误或噪声尖峰。
    4. 连续性检查：如果有效帧太少，返回空数组。

    参数:
    audio: 音频波形 (numpy array, mono)
    sr: 采样率
    fmin, fmax: 人声基频搜索范围 (Hz)
    rms_threshold_ratio: RMS 阈值比例 (相对于最大 RMS 的百分比，用于切除静音)
    prob_threshold: voiced_probability 的阈值 (0.0-1.0)，越高越严格
    min_voiced_frames: 最少保留的有效帧数，否则视为提取失败

    返回:
    f0_clean, time_clean, amp_clean (对齐后的 numpy 数组)
    """


    # 1. 参数设置
    frame_length = 2048
    hop_length = 256

    # 2. 提取基频 (F0) 和 置信度
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=np.nan,  # 显式填充 NaN
    )

    # 3. 提取振幅包络 (RMS Energy)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # 4. 生成时间轴
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # ================= 数据清理核心逻辑 =================

    # Step A: 初始化全 True 掩码
    valid_mask = np.ones(len(f0), dtype=bool)

    # Step B: [RMS 能量过滤] 去除静音
    # 计算动态阈值：最大 RMS 的 x%
    rms_max = np.max(rms)
    if rms_max > 0:
        rms_thresh = rms_max * rms_threshold_ratio
        mask_rms = rms >= rms_thresh
        valid_mask &= mask_rms
    else:
        # 如果整体都没声音，直接返回空
        return np.array([]), np.array([]), np.array([])

    # Step C: [置信度过滤] 去除低置信度帧
    # 即使 voiced_flag 为 True，如果概率太低也视为不可靠
    mask_prob = voiced_probs >= prob_threshold
    valid_mask &= mask_prob

    # Step D: [频率范围二次确认] 去除超出物理范围的异常值
    # pyin 有时会在边界产生跳变，这里再次强制约束
    mask_freq = (f0 >= fmin) & (f0 <= fmax)
    valid_mask &= mask_freq

    # Step E: [NaN 处理] 确保没有 NaN 漏网
    mask_nan = ~np.isnan(f0)
    valid_mask &= mask_nan

    # Step F: [形态学平滑 - 可选] 填补极短的空洞
    # 如果中间只有 1 帧被误删，且前后都是有效帧，可以尝试填补（视需求开启）
    # 这里为了严谨，暂不开启自动填补，避免引入假数据。
    # 如果需要，可以使用 scipy.ndimage.binary_closing 处理 valid_mask

    # 应用掩码
    f0_clean = f0[valid_mask]
    amp_clean = rms[valid_mask]
    time_clean = times[valid_mask]

    # 5. 最终有效性检查
    if len(f0_clean) < min_voiced_frames:
        print(f"警告：清洗后有效帧数 ({len(f0_clean)}) 少于阈值 ({min_voiced_frames})，可能音频太短、太轻或噪声太大。")
        return np.array([]), np.array([]), np.array([])

    # 6. (可选) 离群点平滑：如果某点的 F0 与相邻点差异过大（如八度跳变），可在此处插值替换
    # 这里做一个简单的中值滤波去噪，窗口很小，只去尖刺
    if len(f0_clean) > 5:
        from scipy.signal import medfilt

        # 中值滤波，kernel_size 必须是奇数
        kernel_size = 3
        if len(f0_clean) % 2 == 0:
            kernel_size = 3  # 确保适配
        else:
            kernel_size = min(5, len(f0_clean) if len(f0_clean) % 2 != 0 else len(f0_clean) - 1)
        if kernel_size % 2 == 0: kernel_size -= 1
        if kernel_size < 3: kernel_size = 3

        f0_clean = medfilt(f0_clean, kernel_size=kernel_size)

    return f0_clean, time_clean, amp_clean


# --- 其他特征提取函数 (保持原有逻辑，但增加空值检查) ---

def extract_vibrato_rate(f0_hz, time_sec, min_rate=4.0, max_rate=9.0):
    if len(f0_hz) < 10: return 0.0
    f0_detrended = remove_trend(f0_hz, window_size=0.5)
    if len(time_sec) < 2: return 0.0
    dt = time_sec[1] - time_sec[0]
    if dt <= 0: dt = 0.01

    n = len(f0_detrended)
    yf = fft(f0_detrended)
    xf = fftfreq(n, dt)[:n // 2]

    mask = (xf >= min_rate) & (xf <= max_rate)
    if not np.any(mask): return 0.0

    power = 2.0 / n * np.abs(yf[0:n // 2])
    valid_power = power[mask]
    valid_freqs = xf[mask]

    peak_idx = np.argmax(valid_power)
    return valid_freqs[peak_idx]


def extract_vibrato_extent(f0_hz, time_sec):
    if len(f0_hz) < 10: return 0.0
    f0_detrended = remove_trend(f0_hz, window_size=0.5)
    f0_center = np.mean(f0_hz)
    f0_reconstructed = f0_center + f0_detrended

    if len(time_sec) < 2: return 0.0
    dt = time_sec[1] - time_sec[0]
    if dt <= 0: dt = 0.01

    distance = max(2, int(0.1 / dt))

    peaks, _ = find_peaks(f0_reconstructed, distance=distance)
    valleys, _ = find_peaks(-f0_reconstructed, distance=distance)

    if len(peaks) < 2 or len(valleys) < 2:
        f0_semitones = 12 * np.log2(f0_hz / 440.0 + 1e-6)
        return 2 * np.std(f0_semitones)

    avg_peak_hz = np.mean(f0_reconstructed[peaks])
    avg_valley_hz = np.mean(f0_reconstructed[valleys])

    if avg_valley_hz <= 0: return 0.0

    full_width_st = 12 * np.log2(avg_peak_hz / avg_valley_hz)
    return full_width_st / 2.0


def extract_amp_mod_depth(f0_hz, amp_env, time_sec):
    if len(amp_env) < 10 or len(f0_hz) != len(amp_env): return 0.0
    rate = extract_vibrato_rate(f0_hz, time_sec)

    mean_amp = np.mean(amp_env)
    if mean_amp == 0: return 0.0

    if len(time_sec) < 2: return np.std(amp_env) / mean_amp
    dt = time_sec[1] - time_sec[0]
    if dt <= 0: dt = 0.01
    distance = max(2, int(0.1 / dt))

    peaks, _ = find_peaks(amp_env, distance=distance)
    valleys, _ = find_peaks(-amp_env, distance=distance)

    if len(peaks) < 2 or len(valleys) < 2:
        amp_detrended = remove_trend(amp_env, window_size=0.5)
        return np.std(amp_detrended) / (mean_amp + 1e-6)

    avg_peak = np.mean(amp_env[peaks])
    avg_valley = np.mean(amp_env[valleys])

    modulation_amplitude = (avg_peak - avg_valley) / 2.0
    return modulation_amplitude / (mean_amp + 1e-6)


def extract_sin_fit_err(f0_hz, time_sec):
    if len(f0_hz) < 10: return 1.0
    f0_detrended = remove_trend(f0_hz, window_size=0.5)

    rate_guess = extract_vibrato_rate(f0_hz, time_sec)
    if rate_guess < 4.0: rate_guess = 6.0

    amp_guess = np.std(f0_detrended) * np.sqrt(2)
    t_fit = time_sec - time_sec[0]

    try:
        popt, _ = curve_fit(
            sine_model, t_fit, f0_detrended,
            p0=[amp_guess, rate_guess, 0, 0],
            bounds=([0, 4, -np.pi, -np.inf], [np.inf, 9, np.pi, np.inf]),
            maxfev=2000,
        )
        fitted_signal = sine_model(t_fit, *popt)
        residuals = f0_detrended - fitted_signal
        rmse = np.sqrt(np.mean(residuals ** 2))
        signal_energy = np.sqrt(np.mean(f0_detrended ** 2))

        if signal_energy == 0: return 0.0
        return rmse / signal_energy
    except RuntimeError:
        return 1.0


def extract_rate_cv(f0_hz, time_sec, min_rate=4.0, max_rate=9.0):
    if len(f0_hz) < 50: return 1.0
    f0_detrended = remove_trend(f0_hz, window_size=0.3)

    zero_crossings = np.where(np.diff(np.signbit(f0_detrended)))[0]
    if len(zero_crossings) < 4: return 1.0

    if len(time_sec) <= zero_crossings[-1]:
        # 防止索引越界
        zero_crossings = zero_crossings[zero_crossings < len(time_sec) - 1]
        if len(zero_crossings) < 4: return 1.0

    half_periods = np.diff(time_sec[zero_crossings])
    full_periods = half_periods[::2] + half_periods[::2]

    if len(full_periods) < 3: return 1.0

    inst_freqs = 1.0 / full_periods
    valid_freqs = inst_freqs[(inst_freqs >= min_rate) & (inst_freqs <= max_rate)]

    if len(valid_freqs) < 3: return 1.0

    return np.std(valid_freqs) / (np.mean(valid_freqs) + 1e-6)


def extract_extent_cv(f0_hz, time_sec):
    if len(f0_hz) < 20: return 1.0
    f0_detrended = remove_trend(f0_hz, window_size=0.5)
    f0_center = np.mean(f0_hz)
    f0_reconstructed = f0_center + f0_detrended

    if len(time_sec) < 2: return 1.0
    dt = time_sec[1] - time_sec[0]
    if dt <= 0: dt = 0.01
    distance = max(2, int(0.1 / dt))

    peaks, _ = find_peaks(f0_reconstructed, distance=distance)
    valleys, _ = find_peaks(-f0_reconstructed, distance=distance)

    if len(peaks) < 3 or len(valleys) < 3: return 1.0

    extents = []
    v_idx = 0
    for p_idx in peaks:
        while v_idx < len(valleys) and valleys[v_idx] < p_idx:
            v_idx += 1
        if v_idx < len(valleys):
            diff = f0_reconstructed[p_idx] - f0_reconstructed[valleys[v_idx]]
            if diff > 0:
                extents.append(diff)

    if len(extents) < 3: return 1.0
    return np.std(extents) / (np.mean(extents) + 1e-6)


def extract_onset_time(f0_hz, amp_env, time_sec, threshold_ratio=0.9):
    if len(f0_hz) < 20 or len(amp_env) < 20: return 0.0
    f0_detrended = remove_trend(f0_hz, window_size=0.5)

    if len(time_sec) < 2: return 0.0
    dt = time_sec[1] - time_sec[0]
    if dt <= 0: dt = 0.01

    window_frames = max(5, int(0.1 / dt))
    if len(f0_detrended) < window_frames: return 0.0

    local_std = np.zeros(len(f0_detrended) - window_frames + 1)
    for i in range(len(local_std)):
        local_std[i] = np.std(f0_detrended[i:i + window_frames])

    if len(local_std) == 0 or np.max(local_std) == 0: return 0.0

    steady_state_val = np.mean(local_std[int(len(local_std) * 0.5):])
    if steady_state_val == 0: return 0.0

    threshold = steady_state_val * threshold_ratio
    onset_indices = np.where(local_std >= threshold)[0]

    if len(onset_indices) == 0:
        return time_sec[-1] - time_sec[0]

    first_onset_idx = onset_indices[0]
    actual_time_idx = min(first_onset_idx, len(time_sec) - 1)

    return max(0.0, time_sec[actual_time_idx] - time_sec[0])


def extract_frequency_pulling_ratio(f0_hz, time_sec, onset_window=0.2):
    if len(f0_hz) < 10: return 0.0

    onset_mask = time_sec <= (time_sec[0] + onset_window)
    if not np.any(onset_mask):
        start_f0 = f0_hz[0]
    else:
        start_f0 = np.mean(f0_hz[onset_mask])

    if start_f0 == 0: return 0.0

    mean_f0 = np.mean(f0_hz)
    return (mean_f0 - start_f0) / start_f0


if __name__ == "__main__":
    # 1. 切换目录与加载音频
    proj_root = os.path.abspath(os.path.join(__file__, "../../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    audio_path = "data/Chest_new0206/#A3-5-A.wav"
    full_path = os.path.join(proj_root, audio_path)

    if not os.path.exists(full_path):
        print(f"❌ 错误：找不到文件 {full_path}")
        print("请检查文件路径是否正确。")
    else:
        from src.data_loader import load_audio

        print(f"📂 正在加载：{audio_path} ...")
        audio, _, sr = load_audio(audio_path)

        # 2. 执行提取与清洗
        print("🔍 正在执行 F0 提取与数据清洗 (RMS + Prob Threshold)...")
        f0_seq, time_seq, amp_seq = extract_f0_and_envelope_librosa(
            audio, sr,
            rms_threshold_ratio=0.05,
            prob_threshold=0.65
        )

        if len(f0_seq) == 0:
            print("💥 提取失败：没有足够的有效数据。请检查音频是否为静音或噪声过大。")
        else:
            print(f"✅ 成功提取 {len(f0_seq)} 个有效帧")
            print(f"   时间跨度：{time_seq[0]:.2f}s - {time_seq[-1]:.2f}s")
            print(f"   平均 F0: {np.mean(f0_seq):.2f} Hz")

            # 3. 计算特征
            rate = extract_vibrato_rate(f0_seq, time_seq)
            extent = extract_vibrato_extent(f0_seq, time_seq)
            amp_depth = extract_amp_mod_depth(f0_seq, amp_seq, time_seq)
            # sin_err, t_fit, real_wave, fit_wave = extract_sin_fit_err(f0_seq, time_seq, return_fit_data=True)
            rate_cv = extract_rate_cv(f0_seq, time_seq)
            extent_cv = extract_extent_cv(f0_seq, time_seq)
            onset_t = extract_onset_time(f0_seq, amp_seq, time_seq)
            pull_ratio = extract_frequency_pulling_ratio(f0_seq, time_seq)

            print("\n📊 特征分析结果:")
            print(f"   Vibrato Rate       : {rate:.2f} Hz")
            print(f"   Vibrato Extent     : {extent:.2f} ST")
            print(f"   Amp Mod Depth      : {amp_depth:.3f} ({amp_depth * 100:.1f}%)")
            # print(f"   Sine Fit Error     : {sin_err:.3f} {'⚠️ 高误差!' if sin_err > 0.5 else '✅ 良好'}")
            print(f"   Rate CV (Stability): {rate_cv:.3f}")
            print(f"   Extent CV (Stability): {extent_cv:.3f}")
            print(f"   Onset Time         : {onset_t:.3f} s")
            print(f"   Freq Pulling Ratio : {pull_ratio:.4f}")

            # 4. 深度可视化
            plt.style.use('seaborn-v0_8-whitegrid')
            fig = plt.figure(figsize=(14, 10))

            # Subplot 1: 原始波形与清洗后的 F0/RMS
            ax1 = plt.subplot(2, 1, 1)
            # 为了显示完整波形，重新加载或切片显示 (这里简单显示音频片段)
            t_audio = np.linspace(0, len(audio) / sr, len(audio))
            # 只画一部分避免太密，或者画整体
            plt.plot(t_audio, audio, alpha=0.3, color='gray', label='Raw Audio')
            plt.scatter(
                time_seq, amp_seq / np.max(amp_seq) * 0.5, c='orange', s=10, label='Cleaned RMS (Scaled)', zorder=5
                )
            plt.title(f"Raw Audio & Cleaned Data Points ({len(f0_seq)} frames kept)", fontsize=12)
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.xlim(time_seq[0] - 0.1, time_seq[-1] + 0.1)

            # Subplot 2: F0 曲线
            ax2 = plt.subplot(2, 1, 2)
            plt.plot(time_seq, f0_seq, 'b-', linewidth=1.5, label='Cleaned F0')
            plt.axhline(np.mean(f0_seq), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(f0_seq):.1f}Hz')
            plt.title(f"F0 Contour (Mean: {np.mean(f0_seq):.1f} Hz, Std: {np.std(f0_seq):.1f})", fontsize=12)
            plt.ylabel("Frequency (Hz)")
            plt.legend()
            plt.xlim(time_seq[0] - 0.1, time_seq[-1] + 0.1)

            plt.tight_layout()
            plt.show()

            # # Subplot 3: 🔬 关键诊断：正弦拟合对比 (解决 Error 0.959 之谜)
            # ax3 = plt.subplot(3, 1, 3)
            # if t_fit is not None:
            #     plt.plot(t_fit, real_wave, 'b-', alpha=0.7, label='Real Fluctuation (Detrended)')
            #     plt.plot(t_fit, fit_wave, 'r--', linewidth=2, label='Sine Fit Model')
            #     plt.title(
            #         f"🔬 Diagnostic: Sine Fit vs Real Wave (Error = {sin_err:.3f})", fontsize=12,
            #         color='red' if sin_err > 0.5 else 'black'
            #         )
            #     plt.ylabel("Normalized Amplitude")
            #     plt.xlabel("Time (s) relative to start")
            #     plt.legend()
            #     # 标注出差异大的地方
            #     diff = np.abs(real_wave - fit_wave)
            #     if len(diff) > 0:
            #         max_diff_idx = np.argmax(diff)
            #         plt.scatter(
            #             [t_fit[max_diff_idx]], [real_wave[max_diff_idx]], c='purple', s=50, zorder=10, marker='x',
            #             label='Max Deviation'
            #             )
            # else:
            #     plt.text(0.5, 0.5, "Fit Failed", ha='center', va='center', transform=ax3.transAxes)
            #
            # plt.tight_layout()
            # plt.show()

            # # 5. 额外诊断图：瞬时频率分布 (查看 Rate CV 的来源)
            # plt.figure(figsize=(8, 4))
            # if len(f0_seq) > 50:
            #     f0_det = remove_trend(f0_seq, 0.3)
            #     zc = np.where(np.diff(np.signbit(f0_det)))[0]
            #     zc = zc[zc < len(time_seq) - 1]
            #     if len(zc) >= 4:
            #         half_p = np.diff(time_seq[zc])
            #         full_p = half_p[::2] + half_p[::2]
            #         inst_freqs = 1.0 / full_p
            #         valid_inst = inst_freqs[(inst_freqs >= 4) & (inst_freqs <= 9)]
            #         if len(valid_inst) > 0:
            #             plt.hist(valid_inst, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            #             plt.axvline(rate, color='red', linestyle='--', linewidth=2, label=f'Main Rate: {rate:.2f}Hz')
            #             plt.title(f"Instantaneous Frequency Distribution (Count: {len(valid_inst)})")
            #             plt.xlabel("Frequency (Hz)")
            #             plt.ylabel("Count")
            #             plt.legend()
            #             plt.tight_layout()
            #             plt.show()