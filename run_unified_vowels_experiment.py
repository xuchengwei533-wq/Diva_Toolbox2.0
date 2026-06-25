import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import librosa
except Exception:
    librosa = None

try:
    import parselmouth as pm
except Exception:
    pm = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.io import wavfile
    from scipy import signal
except Exception:
    wavfile = None
    signal = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
except Exception:
    stats = None
    pdist = None
    squareform = None

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.manifold import MDS, Isomap
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
except Exception:
    RidgeCV = None
    MDS = None
    Isomap = None
    LeaveOneGroupOut = None
    cross_val_predict = None
    mean_absolute_error = None
    r2_score = None
    StandardScaler = None


BASE_FEATURE_ALIASES = {
    "H1H2": "H1H2_output",
    "QValue": "Q1",
}

FIVE_VOWELS = ["a", "e", "i", "o", "u"]


def canonical_state_key(text):
    stem = Path(str(text)).stem
    token = stem.split("__")[-1]
    token = re.sub(r"wav$", "", token, flags=re.IGNORECASE)
    if token.startswith("##"):
        token = token[2:]
    return token


def parse_token(token):
    token = re.sub(r"wav$", "", str(token), flags=re.IGNORECASE)
    token = token[2:] if token.startswith("##") else token
    match = re.match(r"(?P<pitch>[#b]*[A-Ga-g]\d)(?:-(?P<score>[135]))?(?:-(?P<variant>[A-Za-z]))?$", token)
    if not match:
        return {
            "state_key": token,
            "pitch_token": None,
            "pitch_class": None,
            "octave": np.nan,
            "score": np.nan,
            "variant": None,
            "state_id": token,
        }
    pitch = match.group("pitch")
    pitch_match = re.match(r"(?P<acc>[#b]*)(?P<letter>[A-Ga-g])(?P<octave>\d)", pitch)
    score_text = match.group("score")
    variant = match.group("variant")
    state_id = f"score_{score_text}{variant or ''}" if score_text else token
    return {
        "state_key": token,
        "pitch_token": pitch,
        "pitch_class": f"{pitch_match.group('acc')}{pitch_match.group('letter').upper()}" if pitch_match else None,
        "octave": int(pitch_match.group("octave")) if pitch_match else np.nan,
        "score": float(score_text) if score_text else np.nan,
        "variant": variant,
        "state_id": state_id,
    }


def maybe_vowel_from_filename(text):
    stem = Path(str(text)).stem.lower()
    patterns = [
        r"(?:^|[_\-\s])vowel[_\-\s]*([aeiou])(?:$|[_\-\s])",
        r"(?:^|[_\-\s])yuan[_\-\s]*([aeiou])(?:$|[_\-\s])",
        r"(?:^|[_\-\s])元音[_\-\s]*([aeiou])(?:$|[_\-\s])",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    return None


def discover_wavs(data_dir):
    rows = []
    for wav_path in sorted(Path(data_dir).rglob("*.wav")):
        rel = wav_path.relative_to(data_dir)
        token = canonical_state_key(wav_path.name)
        parsed = parse_token(token)
        rows.append(
            {
                "wav_path": str(wav_path),
                "relative_path": str(rel),
                "audio_filename": str(rel).replace("\\", "__").replace("/", "__"),
                "source_folder": rel.parts[0] if len(rel.parts) > 1 else "",
                "filename": wav_path.name,
                "explicit_vowel": maybe_vowel_from_filename(wav_path.name),
                **parsed,
            }
        )
    return pd.DataFrame(rows)


def load_cached_features(path):
    df = pd.read_csv(path)
    if "audio_filename" not in df.columns:
        df = df.rename(columns={df.columns[0]: "audio_filename"})
    df = df.rename(columns=BASE_FEATURE_ALIASES)
    df["state_key"] = df["audio_filename"].map(canonical_state_key)
    parsed_rows = [parse_token(x) for x in df["state_key"]]
    parsed = pd.DataFrame(parsed_rows)
    keep_cols = [c for c in parsed.columns if c not in df.columns or c == "state_key"]
    df = pd.concat([df.drop(columns=[c for c in keep_cols if c in df.columns], errors="ignore"), parsed[keep_cols]], axis=1)
    return df


def robust_median(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def robust_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else np.nan


def load_audio_mono(path, target_sr=None):
    if sf is not None:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    elif wavfile is not None:
        sr, audio = wavfile.read(path)
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
    else:
        raise RuntimeError("No WAV reader is available. Install soundfile or scipy.")
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr is not None and librosa is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    elif target_sr is not None and signal is not None and sr != target_sr:
        target_len = int(round(len(audio) * target_sr / float(sr)))
        audio = signal.resample(audio, target_len).astype(np.float32)
        sr = target_sr
    if librosa is not None:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    return np.asarray(audio, dtype=np.float32), int(sr)


def select_stable_center(audio, sr, max_sec=2.0):
    if audio.size == 0:
        return audio
    max_samples = int(max_sec * sr)
    if audio.size <= max_samples:
        return audio
    frame = int(0.05 * sr)
    hop = max(1, frame // 2)
    if frame <= 0 or audio.size < frame:
        start = max(0, audio.size // 2 - max_samples // 2)
        return audio[start:start + max_samples]
    rms = []
    starts = []
    for start in range(0, audio.size - frame + 1, hop):
        seg = audio[start:start + frame]
        rms.append(float(np.sqrt(np.mean(seg * seg))))
        starts.append(start)
    rms = np.asarray(rms)
    starts = np.asarray(starts)
    if rms.size == 0:
        start = max(0, audio.size // 2 - max_samples // 2)
        return audio[start:start + max_samples]
    target_frames = max(1, int(max_samples / hop))
    best = None
    for i in range(0, max(1, rms.size - target_frames + 1)):
        window = rms[i:i + target_frames]
        if window.size == 0:
            continue
        score = robust_std(20.0 * np.log10(window + 1e-12)) - 0.05 * robust_median(window)
        candidate = (score, i)
        if best is None or candidate < best:
            best = candidate
    start = int(starts[best[1]]) if best is not None else max(0, audio.size // 2 - max_samples // 2)
    return audio[start:min(audio.size, start + max_samples)]


def formant_stats(audio, sr, f0_median):
    out = {}
    if pm is None or audio.size == 0:
        return out
    try:
        snd = pm.Sound(audio, sampling_frequency=sr)
        max_formant = min(5500.0, 0.45 * sr)
        formant = pm.praat.call(snd, "To Formant (burg)", 0.01, 5.0, max_formant, 0.025, 50.0)
        n_frames = int(pm.praat.call(formant, "Get number of frames"))
        values = {f"F{i}_hz": [] for i in range(1, 4)}
        bws = {f"BW{i}_hz": [] for i in range(1, 4)}
        for frame_idx in range(1, n_frames + 1):
            t = pm.praat.call(formant, "Get time from frame number", frame_idx)
            for i in range(1, 4):
                f = pm.praat.call(formant, "Get value at time", i, t, "Hertz", "Linear")
                bw = pm.praat.call(formant, "Get bandwidth at time", i, t, "Hertz", "Linear")
                if np.isfinite(f) and 100.0 <= f <= max_formant:
                    values[f"F{i}_hz"].append(float(f))
                if np.isfinite(bw) and 10.0 <= bw <= 3000.0:
                    bws[f"BW{i}_hz"].append(float(bw))
        for name, vals in values.items():
            out[f"{name}_median"] = robust_median(vals)
            out[f"{name}_std"] = robust_std(vals)
        for name, vals in bws.items():
            out[f"{name}_median"] = robust_median(vals)
        if np.isfinite(f0_median) and f0_median > 0:
            for i in range(1, 4):
                f_med = out.get(f"F{i}_hz_median", np.nan)
                if np.isfinite(f_med):
                    harmonic = max(1.0, round(f_med / f0_median))
                    dist_cents = 1200.0 * math.log2((f_med + 1e-12) / (harmonic * f0_median + 1e-12))
                    out[f"F{i}_nearest_harmonic_distance_cents"] = float(abs(dist_cents))
                    out[f"F{i}_to_F0_ratio"] = float(f_med / f0_median)
    except Exception as exc:
        out["formant_error"] = str(exc)
    return out


def extract_audio_features(wav_df, limit=None, include_formants=False):
    if librosa is None or (sf is None and wavfile is None):
        return pd.DataFrame()
    rows = []
    work = wav_df.head(limit).copy() if limit else wav_df.copy()
    for idx, row in work.iterrows():
        wav_path = row["wav_path"]
        try:
            audio, sr = load_audio_mono(wav_path)
            audio = select_stable_center(audio, sr)
            duration = audio.size / float(sr) if sr > 0 else np.nan
            rms = librosa.feature.rms(y=audio)[0]
            rms_db = 20.0 * np.log10(rms + 1e-12)
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            f0_valid = f0[np.isfinite(f0)] if f0 is not None else np.array([])
            f0_median = robust_median(f0_valid)
            f0_cents = 1200.0 * np.log2((f0_valid + 1e-12) / (f0_median + 1e-12)) if np.isfinite(f0_median) and f0_median > 0 else np.array([])
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            flatness = librosa.feature.spectral_flatness(y=audio)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            feat = {
                "state_key": row["state_key"],
                "duration_sec": float(duration),
                "rms_db_median": robust_median(rms_db),
                "rms_db_std": robust_std(rms_db),
                "f0_hz_median": f0_median,
                "f0_cents_std": robust_std(f0_cents),
                "voiced_ratio": float(np.mean(np.isfinite(f0))) if f0 is not None and len(f0) else np.nan,
                "spectral_centroid_median": robust_median(centroid),
                "spectral_bandwidth_median": robust_median(bandwidth),
                "spectral_rolloff85_median": robust_median(rolloff),
                "spectral_flatness_median": robust_median(flatness),
                "zero_crossing_rate_median": robust_median(zcr),
            }
            for i in range(mfcc.shape[0]):
                feat[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
                feat[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))
            if include_formants:
                feat.update(formant_stats(audio, sr, f0_median))
            rows.append(feat)
            print(f"[audio] {idx + 1}/{len(work)} {Path(wav_path).name}", flush=True)
        except Exception as exc:
            rows.append({"state_key": row["state_key"], "audio_feature_error": str(exc)})
            print(f"[audio][error] {Path(wav_path).name}: {exc}", flush=True)
    return pd.DataFrame(rows)


def numeric_feature_columns(df):
    blocked = {
        "score",
        "octave",
        "group_size",
    }
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= max(5, int(0.2 * len(df))) and values.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def zscore_matrix(df, feature_cols):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0).replace(0, 1.0)
    Z = (X - means) / stds
    return Z.to_numpy(dtype=float), means, stds


def run_pca(Z, n_components=5):
    Zc = Z - np.mean(Z, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    n = max(1, Z.shape[0] - 1)
    eig = (s ** 2) / n
    total = float(np.sum(eig))
    ratio = eig / total if total > 0 else np.zeros_like(eig)
    coords = u[:, :n_components] * s[:n_components]
    loadings = vt[:n_components, :].T
    return coords, ratio[:n_components], loadings


def add_pitch_residual_features(df, feature_cols):
    out = df.copy()
    for col in feature_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        med = values.median()
        values = values.fillna(med)
        group_mean = values.groupby(out["pitch_token"]).transform("mean")
        out[f"resid_{col}"] = values - group_mean
    return out


def euclidean_distances(X):
    if pdist is not None and squareform is not None:
        return squareform(pdist(X, metric="euclidean"))
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def compactness_by_group(df, coord_cols, group_col):
    rows = []
    for name, sub in df.dropna(subset=coord_cols).groupby(group_col):
        X = sub[coord_cols].to_numpy(dtype=float)
        if len(sub) < 2:
            continue
        centroid = X.mean(axis=0)
        d_centroid = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        d_pair = euclidean_distances(X)
        pair_values = d_pair[np.triu_indices_from(d_pair, k=1)]
        rows.append(
            {
                group_col: name,
                "n": len(sub),
                "centroid_radius_mean": float(np.mean(d_centroid)),
                "centroid_radius_std": float(np.std(d_centroid)),
                "pairwise_distance_mean": float(np.mean(pair_values)),
                "pairwise_distance_std": float(np.std(pair_values)),
                "pairwise_distance_cv": float(np.std(pair_values) / (np.mean(pair_values) + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def distance_to_score_centroid(df, coord_cols):
    out = df.copy()
    coords = out[coord_cols].to_numpy(dtype=float)
    out["distance_to_own_score_centroid"] = np.nan
    out["distance_to_high_score_centroid"] = np.nan
    high_mask = out["score"] == out["score"].max()
    high_centroid = coords[high_mask].mean(axis=0) if high_mask.sum() else None
    for score, idx in out.groupby("score").groups.items():
        idx = np.asarray(list(idx))
        if len(idx) == 0 or not np.isfinite(score):
            continue
        centroid = coords[idx].mean(axis=0)
        out.loc[out.index[idx], "distance_to_own_score_centroid"] = np.sqrt(np.sum((coords[idx] - centroid) ** 2, axis=1))
        if high_centroid is not None:
            out.loc[out.index[idx], "distance_to_high_score_centroid"] = np.sqrt(np.sum((coords[idx] - high_centroid) ** 2, axis=1))
    return out


def pairwise_state_matrix(df, coord_cols):
    rows = []
    labels = sorted([x for x in df["state_id"].dropna().unique()], key=str)
    for a in labels:
        for b in labels:
            vals = []
            for _pitch, sub in df.groupby("pitch_token"):
                aa = sub[sub["state_id"] == a]
                bb = sub[sub["state_id"] == b]
                if len(aa) == 1 and len(bb) == 1:
                    va = aa[coord_cols].to_numpy(dtype=float)[0]
                    vb = bb[coord_cols].to_numpy(dtype=float)[0]
                    vals.append(float(np.sqrt(np.sum((va - vb) ** 2))))
            rows.append({"state_a": a, "state_b": b, "mean_distance": robust_median(vals), "n_pairs": len(vals)})
    return pd.DataFrame(rows)


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if stats is None:
        xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
        xr = xr - np.mean(xr)
        yr = yr - np.mean(yr)
        denom = math.sqrt(float(np.sum(xr * xr) * np.sum(yr * yr)))
        return (float(np.sum(xr * yr) / denom) if denom > 0 else np.nan), np.nan
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def permutation_compactness_test(df, metric_col, score_col="score", n_perm=5000, seed=2026):
    clean = df[[metric_col, score_col]].dropna().copy()
    if clean[score_col].nunique() < 2:
        return {}
    low_score = clean[score_col].min()
    high_score = clean[score_col].max()
    low = clean.loc[clean[score_col] == low_score, metric_col].to_numpy(dtype=float)
    high = clean.loc[clean[score_col] == high_score, metric_col].to_numpy(dtype=float)
    if len(low) < 2 or len(high) < 2:
        return {}
    observed = float(np.mean(low) - np.mean(high))
    rng = np.random.default_rng(seed)
    values = clean[metric_col].to_numpy(dtype=float)
    scores = clean[score_col].to_numpy()
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(scores)
        diff = np.mean(values[shuffled == low_score]) - np.mean(values[shuffled == high_score])
        if diff >= observed:
            count += 1
    return {
        "metric": metric_col,
        "low_score": low_score,
        "high_score": high_score,
        "low_mean": float(np.mean(low)),
        "high_mean": float(np.mean(high)),
        "observed_low_minus_high": observed,
        "one_sided_p_low_greater_than_high": float((count + 1) / (n_perm + 1)),
        "n_perm": int(n_perm),
    }


def predictive_validation(df, feature_cols, out_dir):
    if RidgeCV is None or LeaveOneGroupOut is None or cross_val_predict is None:
        return pd.DataFrame()
    clean = df.dropna(subset=["score", "pitch_token"]).copy()
    clean = clean[np.isfinite(clean["score"])]
    if clean["score"].nunique() < 2 or clean["pitch_token"].nunique() < 3:
        return pd.DataFrame()
    X, _, _ = zscore_matrix(clean, feature_cols)
    y = clean["score"].to_numpy(dtype=float)
    groups = clean["pitch_token"].to_numpy()
    logo = LeaveOneGroupOut()
    model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    try:
        pred = cross_val_predict(model, X, y, groups=groups, cv=logo)
    except Exception as exc:
        return pd.DataFrame([{"model": "ridge_leave_pitch_out", "error": str(exc)}])
    rho, p = spearman(pred, y)
    result = pd.DataFrame(
        [
            {
                "model": "ridge_leave_pitch_out",
                "n": len(clean),
                "n_pitch_groups": clean["pitch_token"].nunique(),
                "mae": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred)),
                "spearman_rho": rho,
                "spearman_p": p,
            }
        ]
    )
    pred_df = clean[["audio_filename", "state_key", "pitch_token", "state_id", "score"]].copy()
    pred_df["predicted_score"] = pred
    pred_df.to_csv(out_dir / "leave_pitch_out_predictions.csv", index=False, encoding="utf-8-sig")
    return result


def plot_pca(df, out_dir, prefix):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scores = sorted(df["score"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / max(1, len(scores) - 1)) for i, s in enumerate(scores)}
    for score in scores:
        sub = df[df["score"] == score]
        ax.scatter(sub["PC1"], sub["PC2"], s=55, color=colors[score], label=f"score {int(score)}", alpha=0.82, edgecolor="white", linewidth=0.6)
    for pitch, sub in df.groupby("pitch_token"):
        if len(sub) >= 2:
            ordered = sub.sort_values(["score", "state_id"])
            ax.plot(ordered["PC1"], ordered["PC2"], color="0.78", linewidth=0.8, alpha=0.55, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Low-dimensional acoustic organization")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pca_scores.png", dpi=220)
    fig.savefig(out_dir / f"{prefix}_pca_scores.svg")
    plt.close(fig)


def plot_distance_by_score(df, metric, out_dir, prefix):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns is not None:
        sns.boxplot(data=df, x="score", y=metric, color="#d8e2dc", ax=ax)
        sns.stripplot(data=df, x="score", y=metric, color="#2f3e46", size=4, jitter=0.2, alpha=0.75, ax=ax)
    else:
        scores = sorted(df["score"].dropna().unique())
        vals = [df.loc[df["score"] == s, metric].dropna().to_numpy() for s in scores]
        ax.boxplot(vals, labels=[str(int(s)) for s in scores])
    ax.set_xlabel("Expert/filename score")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Acoustic distance by unity score")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{metric}.png", dpi=220)
    fig.savefig(out_dir / f"{prefix}_{metric}.svg")
    plt.close(fig)


def plot_pairwise_heatmap(matrix_df, out_dir):
    if plt is None or matrix_df.empty:
        return
    pivot = matrix_df.pivot(index="state_a", columns="state_b", values="mean_distance")
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns is not None:
        sns.heatmap(pivot, cmap="mako", annot=True, fmt=".2f", square=True, ax=ax)
    else:
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        fig.colorbar(im, ax=ax)
    ax.set_title("Pitch-matched state distance matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "pitch_matched_state_distance_heatmap.png", dpi=220)
    fig.savefig(out_dir / "pitch_matched_state_distance_heatmap.svg")
    plt.close(fig)


def write_protocol(out_dir, mode_note, feature_cols, pca_ratio, compact_test, validation):
    pca_text = ", ".join([f"PC{i + 1}={v:.1%}" for i, v in enumerate(pca_ratio[:3])])
    validation_text = "not available"
    if validation is not None and not validation.empty:
        row = validation.iloc[0].to_dict()
        validation_text = ", ".join([f"{k}={v:.4g}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else f"{k}={v}" for k, v in row.items()])
    compact_text = json.dumps(compact_test, ensure_ascii=False, indent=2) if compact_test else "not available"
    text = f"""# Unified Vowels Low-Dimensional Organization Experiment

## Current Data Mode

{mode_note}

## Experimental Logic

1. Build an acoustic feature table for every token.
2. Standardize all acoustic features and create a low-dimensional representation with PCA.
3. Remove pitch-block effects when comparing filename/expert scores, so score effects are not just pitch effects.
4. Measure organization rather than a single parameter: centroid radius, pairwise distance, distance to score centroids, and pitch-matched distance matrices.
5. Validate whether the acoustic organization predicts the score under leave-one-pitch-out validation.

## Feature Blocks

The analysis used {len(feature_cols)} numeric features:

{", ".join(feature_cols)}

## PCA Variance

{pca_text}

## Main Compactness Test

```json
{compact_text}
```

## Predictive Validation

{validation_text}

## How This Maps To A Full Five-Vowel Paper

For the final paper dataset, add a metadata CSV with at least:

- `audio_filename` or `wav_path`
- `sample_id` for the singer/task token
- `vowel` as one of `/a e i o u/`
- `unity_score` from expert teachers
- optional `singer_id`, `pitch`, `register`, `teacher_id`

Then the same framework should compute five-vowel compactness per `sample_id` and regress expert `unity_score` on the cross-vowel structural metrics.
"""
    (out_dir / "experiment_protocol.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run low-dimensional acoustic organization analysis for unified vowels.")
    parser.add_argument("--data-dir", default="data/Unify", help="Folder containing WAV files.")
    parser.add_argument("--cached-feats", default="outputs/Unify/feats_data.csv", help="Existing cached feature table.")
    parser.add_argument("--out-dir", default="outputs/Unify/unified_vowels_experiment", help="Output folder.")
    parser.add_argument("--skip-audio", action="store_true", help="Only use cached feature table.")
    parser.add_argument("--extract-audio-only", action="store_true", help="Extract extended audio features and exit before PCA/statistics.")
    parser.add_argument("--include-formants", action="store_true", help="Also extract Praat/Burg formant features. Slower.")
    parser.add_argument("--no-existing-audio", action="store_true", help="Do not merge an existing extended_audio_features.csv when --skip-audio is used.")
    parser.add_argument("--audio-limit", type=int, default=None, help="Optional debug limit for audio feature extraction.")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_df = discover_wavs(args.data_dir)
    cached = load_cached_features(args.cached_feats)
    cached_features = cached.drop(columns=[c for c in ["audio_filename"] if c in cached.columns]).copy()

    merged = cached.copy()
    wav_cols = ["state_key", "wav_path", "relative_path", "source_folder", "filename", "explicit_vowel"]
    if not wav_df.empty:
        merged = merged.merge(wav_df[wav_cols], on="state_key", how="left", suffixes=("", "_wav"))

    mode_note = (
        "No explicit five-vowel labels were detected in the current folder. "
        "This run is a proxy experiment on the existing `Unify` same-pitch score-state data: "
        "`pitch_token` is treated as the blocking group and filename score states are treated as the available semantic labels. "
        "Do not describe these outputs as final /a e i o u/ evidence until a vowel-labeled metadata file is added."
    )
    if merged.get("explicit_vowel", pd.Series(dtype=object)).isin(FIVE_VOWELS).any():
        mode_note = "Explicit vowel labels were detected and used as unit labels."

    audio_features = pd.DataFrame()
    if not args.skip_audio and not wav_df.empty and librosa is not None and (sf is not None or wavfile is not None):
        audio_features = extract_audio_features(wav_df, limit=args.audio_limit, include_formants=args.include_formants)
        audio_features.to_csv(out_dir / "extended_audio_features.csv", index=False, encoding="utf-8-sig")
        if args.extract_audio_only:
            summary = {
                "data_dir": str(Path(args.data_dir).resolve()),
                "out_dir": str(out_dir.resolve()),
                "n_wav_files": int(len(wav_df)),
                "n_audio_feature_rows": int(len(audio_features)),
                "audio_features_file": str((out_dir / "extended_audio_features.csv").resolve()),
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
    elif args.extract_audio_only:
        raise RuntimeError("Audio extraction requested, but librosa/soundfile are not available in this Python environment.")

    existing_audio_path = out_dir / "extended_audio_features.csv"
    if audio_features.empty and not args.no_existing_audio and existing_audio_path.exists():
        audio_features = pd.read_csv(existing_audio_path)

    if not audio_features.empty:
        merged = merged.merge(audio_features, on="state_key", how="left", suffixes=("", "_audio"))

    merged["state_id"] = merged["state_id"].fillna(merged["state_key"])
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["octave"] = pd.to_numeric(merged["octave"], errors="coerce")
    merged["group_size"] = merged.groupby("pitch_token")["state_key"].transform("count")
    merged.to_csv(out_dir / "analysis_feature_table.csv", index=False, encoding="utf-8-sig")

    feature_cols = numeric_feature_columns(merged)
    if len(feature_cols) < 2:
        raise ValueError("Not enough numeric features for low-dimensional analysis.")
    Z, means, stds = zscore_matrix(merged, feature_cols)
    coords, pca_ratio, loadings = run_pca(Z, n_components=min(5, len(feature_cols), len(merged)))
    for i in range(coords.shape[1]):
        merged[f"PC{i + 1}"] = coords[:, i]

    loading_df = pd.DataFrame(loadings, index=feature_cols, columns=[f"PC{i + 1}" for i in range(loadings.shape[1])])
    loading_df["feature_mean"] = means
    loading_df["feature_std"] = stds
    loading_df.to_csv(out_dir / "pca_loadings.csv", encoding="utf-8-sig")
    pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(pca_ratio))], "explained_variance_ratio": pca_ratio}).to_csv(
        out_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig"
    )

    resid_df = add_pitch_residual_features(merged, feature_cols)
    resid_cols = [f"resid_{c}" for c in feature_cols]
    R, _, _ = zscore_matrix(resid_df, resid_cols)
    rcoords, rpca_ratio, _ = run_pca(R, n_components=min(5, R.shape[1], len(resid_df)))
    for i in range(rcoords.shape[1]):
        resid_df[f"rPC{i + 1}"] = rcoords[:, i]

    coord_cols = [c for c in ["PC1", "PC2", "PC3"] if c in merged.columns]
    rcoord_cols = [c for c in ["rPC1", "rPC2", "rPC3"] if c in resid_df.columns]
    merged = distance_to_score_centroid(merged, coord_cols)
    resid_df = distance_to_score_centroid(resid_df, rcoord_cols)

    pitch_compact = compactness_by_group(merged, coord_cols, "pitch_token")
    score_compact = compactness_by_group(resid_df, rcoord_cols, "score")
    state_matrix = pairwise_state_matrix(merged, coord_cols)

    pitch_compact.to_csv(out_dir / "pitch_group_compactness.csv", index=False, encoding="utf-8-sig")
    score_compact.to_csv(out_dir / "score_compactness_after_pitch_residualization.csv", index=False, encoding="utf-8-sig")
    state_matrix.to_csv(out_dir / "pitch_matched_state_distance_matrix.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "pca_token_coordinates.csv", index=False, encoding="utf-8-sig")
    resid_df.to_csv(out_dir / "pitch_residual_pca_token_coordinates.csv", index=False, encoding="utf-8-sig")

    metric = "distance_to_own_score_centroid"
    compact_test = permutation_compactness_test(resid_df, metric, n_perm=args.permutations)
    pd.DataFrame([compact_test]).to_csv(out_dir / "permutation_compactness_test.csv", index=False, encoding="utf-8-sig")

    correlation_rows = []
    for col in ["distance_to_own_score_centroid", "distance_to_high_score_centroid", "rPC1", "rPC2", "rPC3"]:
        if col in resid_df.columns:
            rho, p = spearman(resid_df[col].to_numpy(dtype=float), resid_df["score"].to_numpy(dtype=float))
            correlation_rows.append({"variable": col, "score_spearman_rho": rho, "p_value": p})
    corr_df = pd.DataFrame(correlation_rows)
    corr_df.to_csv(out_dir / "score_correlations.csv", index=False, encoding="utf-8-sig")

    validation = predictive_validation(resid_df, resid_cols, out_dir)
    validation.to_csv(out_dir / "predictive_validation.csv", index=False, encoding="utf-8-sig")

    if MDS is not None:
        try:
            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=2026, normalized_stress="auto")
            mds_coords = mds.fit_transform(Z)
            mds_df = merged[["audio_filename", "state_key", "pitch_token", "state_id", "score"]].copy()
            mds_df["MDS1"] = mds_coords[:, 0]
            mds_df["MDS2"] = mds_coords[:, 1]
            mds_df.to_csv(out_dir / "mds_token_coordinates.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            (out_dir / "mds_error.txt").write_text(str(exc), encoding="utf-8")
    if Isomap is not None:
        try:
            n_neighbors = min(8, max(2, len(merged) // 8))
            iso = Isomap(n_components=2, n_neighbors=n_neighbors)
            iso_coords = iso.fit_transform(Z)
            iso_df = merged[["audio_filename", "state_key", "pitch_token", "state_id", "score"]].copy()
            iso_df["ISOMAP1"] = iso_coords[:, 0]
            iso_df["ISOMAP2"] = iso_coords[:, 1]
            iso_df.to_csv(out_dir / "isomap_token_coordinates.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            (out_dir / "isomap_error.txt").write_text(str(exc), encoding="utf-8")

    plot_pca(merged, out_dir, "all_features")
    if metric in resid_df.columns:
        plot_distance_by_score(resid_df, metric, out_dir, "pitch_residual")
    plot_pairwise_heatmap(state_matrix, out_dir)

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "cached_features": str(Path(args.cached_feats).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_wav_files": int(len(wav_df)),
        "n_tokens": int(len(merged)),
        "n_pitch_groups": int(merged["pitch_token"].nunique(dropna=True)),
        "n_numeric_features": int(len(feature_cols)),
        "audio_features_extracted": bool(not audio_features.empty),
        "mode_note": mode_note,
        "pca_explained_variance_first3": [float(x) for x in pca_ratio[:3]],
        "pitch_residual_pca_explained_variance_first3": [float(x) for x in rpca_ratio[:3]],
        "compactness_test": compact_test,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_protocol(out_dir, mode_note, feature_cols, pca_ratio, compact_test, validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
