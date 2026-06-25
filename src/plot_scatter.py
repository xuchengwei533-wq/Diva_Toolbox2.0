import os
import sys
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
try:
    from PIL import Image
except Exception:
    Image = None
try:
    from scipy.stats import gaussian_kde
    from scipy.stats import t as student_t
except Exception:
    gaussian_kde = None
    student_t = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_parser import parse_label, parse_suffix_type, parse_pitch_digit


# ==============================================================================
# 配置与常量
# ==============================================================================

# Internal subset combinations. Display labels are Pressed phonation,
# Breathy phonation, and ALL.
SUBSET_GROUPS = [
    ["A", "1"],
    ["B", "1"],
    ["A", "B", "1"]  # All
]
# 绘图相关常量
PLOT_JITTER_RANGE = 0.12
FIG_SIZE = (16, 6)
DPI = 100
ALPHA = 0.8
MARKER_SIZE = 36
SAVE_PNG = True
SAVE_JPG = False

SCORE_TO_CLASS = {
    1: "A",
    3: "B",
    5: "C",
}
CLASS_TO_SCORE = {
    "A": 1,
    "B": 3,
    "C": 5,
}
CLASS_COLOR_MAP = {
    "A": "#F29AA0",
    "B": "#A895E0",
    "C": "#8ED0F0",
}
CLASS_ORDER = ["A", "B", "C"]
LEGACY_FEATURE_ALIASES = {
    "QValue": "Q1",
    "H1H2": "H1H2_output",
}
SUBSET_TITLES = ["Pressed phonation", "Breathy phonation", "ALL"]
PAPER_DPI = 300


# ==============================================================================
# 数据处理工具函数
# ==============================================================================

def enrich_df_with_metadata(df_feats_stats: pd.DataFrame) -> pd.DataFrame:
    """
    在输入的 DataFrame 中新增元数据列：'score_label', 'subset_type', 'pitch_digit'。
    这些列基于行索引 (文件名) 解析得出。

    Args:
        df_feats_stats: 索引为文件名的特征统计 DataFrame。

    Returns:
        增加了元数据列的 DataFrame 副本。
    """
    df_enriched = df_feats_stats.copy()
    filenames = df_enriched.index
    # 向量化/Map 操作添加元数据列
    df_enriched['score_label'] = filenames.map(parse_label)
    df_enriched['subset_type'] = filenames.map(parse_suffix_type)
    df_enriched['pitch_digit'] = filenames.map(parse_pitch_digit)
    df_enriched["score_class"] = df_enriched["score_label"].map(score_to_class_label)
    return df_enriched


def filter_df_by_subset(
        df_enriched: pd.DataFrame,
        allowed_subsets: list = None
) -> pd.DataFrame:
    """
    根据允许的子集标签过滤 DataFrame。

    Args:
        df_enriched: 已经包含 'subset_type' 列的 DataFrame。
        allowed_subsets: 允许保留的子集标签列表 (e.g., ['A', '1'])。

    Returns:
        过滤后的 DataFrame。
    """
    # 若 allowed_subsets 为空或 None，则返回原始 DataFrame 的副本，不进行过滤
    if not allowed_subsets:
        return df_enriched.copy()
    # 构建过滤掩码并应用过滤
    mask = df_enriched['subset_type'].isin(allowed_subsets)
    filtered = df_enriched[mask].copy()
    if filtered.empty:
        print(f"[!] 警告：筛选子集 {allowed_subsets} 后无剩余数据。")
    return filtered


def remove_outlier_jitter_df(df: pd.DataFrame, jitter_col: str = "Jitter") -> pd.DataFrame:
    """
    从 DataFrame 中找到 jitter 特征值最大的行，将其移除。

    Args:
        df: 特征统计 DataFrame。
        jitter_col: Jitter 列的名称。

    Returns:
        移除异常值后的 DataFrame 副本。
    """
    if jitter_col not in df.columns:
        print(f"[!] 列 '{jitter_col}' 不存在，无法移除异常值。")
        return df

    # 找到最大值的索引
    max_idx = df[jitter_col].idxmax()
    max_val = df.loc[max_idx, jitter_col]

    if pd.isna(max_val):
        print("[*] Jitter 列全为 NaN，无需移除。")
        return df

    print(f"[!] 移除异常值：{max_idx}，{jitter_col}={max_val:.6f}")

    # 返回删除了该行的副本
    return df.drop(index=max_idx)


def normalize_feature_naming(df: pd.DataFrame) -> pd.DataFrame:
    """
    将历史特征命名映射到当前标准命名，确保绘图流程兼容旧产物。
    """
    out = df.copy()
    for old_name, new_name in LEGACY_FEATURE_ALIASES.items():
        if old_name not in out.columns:
            continue
        if new_name in out.columns:
            out[new_name] = out[new_name].combine_first(out[old_name])
            out = out.drop(columns=[old_name])
        else:
            out = out.rename(columns={old_name: new_name})
    return out


def score_to_class_label(score):
    """
    Convert numeric score labels to ordinal class labels:
    1 -> A, 3 -> B, 5 -> C.
    Return None for invalid values.
    """
    try:
        s = int(score)
    except Exception:
        return None
    return SCORE_TO_CLASS.get(s)


# ==============================================================================
# 绘图工具函数
# ==============================================================================

def create_legend_handles_score(ax):
    """兼容旧接口：返回 A/B/C 类别图例句柄。"""
    return create_legend_handles_class(ax)


def create_legend_handles_class(ax=None):
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cls,
            markerfacecolor=CLASS_COLOR_MAP[cls],
            markeredgecolor="k",
            markeredgewidth=0.4,
            markersize=8,
        )
        for cls in CLASS_ORDER
    ]
    return handles


def _compute_axis_limits(values, pad_ratio=0.05):
    """
    基于输入数值计算统一坐标轴范围，并添加少量边距，避免点贴边显示。
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        base = abs(vmin) if abs(vmin) > 1e-12 else 1.0
        pad = base * pad_ratio
    else:
        pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


def _feature_file_token(name):
    """
    用于文件名的特征标识，避免出现旧列名或复杂符号。
    """
    mapping = {
        "H1H2_output": "H1H2",
        "H1H2": "H1H2",
        "CPP": "CPP",
        "Q1": "Q1",
        "QValue": "Q1",
        "HNR": "HNR",
        "SpectralSlope": "SpectralSlope",
        "LowFreqEnergyRatio": "LowFreqEnergyRatio",
        "HighFreqNoiseRatio": "HF residual-noise ratio",
        "Jitter": "Jitter",
        "Shimmer": "Shimmer",
    }
    token = mapping.get(name, str(name))
    token = token.replace(" ", "-")
    return token


def _save_figure(
        fig,
        out_dir,
        stem,
        dpi=300,
        save_png=SAVE_PNG,
        save_jpg=SAVE_JPG,
        bbox_tight=True,
        pad_inches=0.05,
):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    bbox = "tight" if bbox_tight else None
    pad = pad_inches if bbox_tight else None
    if save_png:
        png_path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches=bbox, pad_inches=pad)
        saved_paths.append(png_path)
    if save_jpg:
        jpg_path = os.path.join(out_dir, f"{stem}.jpg")
        fig.savefig(jpg_path, dpi=dpi, bbox_inches=bbox, pad_inches=pad)
        saved_paths.append(jpg_path)
    return saved_paths


def _plot_1d_scatter(ax, coords, scores, pitches_arr, feat_tuple, is_last_subset):
    """
    Plot 1D scatter with score classes A/B/C on the x-axis.
    Pitch/register information is intentionally ignored.
    """
    _ = pitches_arr
    y_coords = coords[:, 0]
    scores = np.asarray(scores, dtype=float)

    x_base = []
    class_labels = []
    for s in scores:
        cls = score_to_class_label(s)
        if cls is None:
            x_base.append(np.nan)
            class_labels.append(None)
        else:
            x_base.append(CLASS_ORDER.index(cls))
            class_labels.append(cls)

    x_base = np.asarray(x_base, dtype=float)
    jitter = np.random.uniform(
        -PLOT_JITTER_RANGE,
        PLOT_JITTER_RANGE,
        size=x_base.shape,
    )
    x_coords = x_base + jitter

    for cls in CLASS_ORDER:
        mask = np.asarray([c == cls for c in class_labels])
        if np.any(mask):
            class_index = CLASS_ORDER.index(cls)
            ax.scatter(
                x_coords[mask],
                y_coords[mask],
                c=CLASS_COLOR_MAP[cls],
                s=MARKER_SIZE,
                alpha=ALPHA,
                edgecolors="k",
                linewidths=0.3,
                label=cls,
            )
            vals = y_coords[mask]
            stats = _summary_stats(vals)
            if stats["n"] > 0:
                ax.hlines(
                    stats["median"],
                    class_index - 0.24,
                    class_index + 0.24,
                    color="#222222",
                    linewidth=1.6,
                    zorder=4,
                )
                if np.isfinite(stats["mean"]):
                    if np.isfinite(stats["ci95_low"]) and np.isfinite(stats["ci95_high"]):
                        yerr = np.array(
                            [
                                [stats["mean"] - stats["ci95_low"]],
                                [stats["ci95_high"] - stats["mean"]],
                            ]
                        )
                    else:
                        yerr = None
                    ax.errorbar(
                        [class_index],
                        [stats["mean"]],
                        yerr=yerr,
                        fmt="D",
                        color="#111111",
                        ecolor="#111111",
                        elinewidth=1.0,
                        capsize=4,
                        markersize=4.5,
                        zorder=5,
                    )
                ax.text(
                    class_index,
                    0.965,
                    f"n={stats['n']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
                )

    ax.set_xlabel("Resonance category")
    ax.set_ylabel(_pretty_feature_label(feat_tuple[0]))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CLASS_ORDER)

    if is_last_subset:
        handles = create_legend_handles_class(ax)
        stat_handles = [
            Line2D([0], [0], marker="D", color="#111111", linestyle="", markersize=5, label="Mean +/- 95% CI"),
            Line2D([0], [0], color="#222222", linewidth=1.6, label="Median"),
        ]
        ax.legend(
            handles=handles + stat_handles,
            title="Category",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )


def _summary_stats(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
    median = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    if n > 1 and np.isfinite(sd):
        crit = float(student_t.ppf(0.975, n - 1)) if student_t is not None else 1.96
        half_width = crit * sd / np.sqrt(n)
        ci95_low = mean - half_width
        ci95_high = mean + half_width
    else:
        ci95_low = np.nan
        ci95_high = np.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "ci95_low": float(ci95_low) if np.isfinite(ci95_low) else np.nan,
        "ci95_high": float(ci95_high) if np.isfinite(ci95_high) else np.nan,
    }


def _summarize_1d_panel(df_panel, feature, panel_title):
    rows = []
    for cls in CLASS_ORDER:
        vals = df_panel.loc[df_panel["score_class"] == cls, feature].values
        stats = _summary_stats(vals)
        row = {
            "panel": panel_title,
            "category": cls,
            "feature": feature,
            "display_feature": _pretty_feature_label(feature),
        }
        row.update(stats)
        rows.append(row)
    return rows


def _plot_2d_scatter(ax, coords, scores, feat_tuple, is_last_subset):
    """
    Plot 2D scatter. Color encodes resonance category A/B/C, and each class
    includes its own covariance ellipse to indicate the distribution region.
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    scores = np.asarray(scores, dtype=float)
    class_labels = [score_to_class_label(s) for s in scores]

    for cls in CLASS_ORDER:
        mask = np.asarray([c == cls for c in class_labels])
        if np.any(mask):
            x_plot, y_plot, used_ellipse = _filter_xy_by_cov_ellipse_2d(
                x_coords[mask],
                y_coords[mask],
                n_std=2.0,
            )
            if x_plot.size == 0:
                continue
            if used_ellipse:
                _add_cov_ellipse_2d(
                    ax,
                    x_plot,
                    y_plot,
                    color=CLASS_COLOR_MAP[cls],
                    n_std=2.0,
                )
            ax.scatter(
                x_plot,
                y_plot,
                c=CLASS_COLOR_MAP[cls],
                s=MARKER_SIZE,
                alpha=ALPHA,
                edgecolors="k",
                linewidths=0.3,
                label=cls,
            )

    ax.set_xlabel(_pretty_feature_label(feat_tuple[0]))
    ax.set_ylabel(_pretty_feature_label(feat_tuple[1]))
    if is_last_subset:
        handles = create_legend_handles_class(ax)
        ax.legend(handles=handles, title="Category", loc="upper right", fontsize=8)


def _plot_3d_scatter(ax, coords, scores, feat_tuple, show_legend=True):
    """
    Plot 3D scatter. Color encodes resonance category A/B/C.
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    scores = np.asarray(scores, dtype=float)
    class_labels = [score_to_class_label(s) for s in scores]

    for cls in CLASS_ORDER:
        mask = np.asarray([c == cls for c in class_labels])
        if np.any(mask):
            ax.scatter(
                x_coords[mask],
                y_coords[mask],
                z_coords[mask],
                c=CLASS_COLOR_MAP[cls],
                s=MARKER_SIZE,
                alpha=ALPHA,
                edgecolors="k",
                linewidths=0.3,
                label=cls,
            )

    ax.set_xlabel(_pretty_feature_label(feat_tuple[0]))
    ax.set_ylabel(_pretty_feature_label(feat_tuple[1]))
    ax.set_zlabel(_pretty_feature_label(feat_tuple[2]))
    if show_legend:
        handles = create_legend_handles_class(ax)
        ax.legend(handles=handles, title="Category", loc="upper left", fontsize=8)


def _clean_numeric_panel_df(df, feat_names):
    """
    面板数据清洗：
    - 确保存在 score_label 与目标特征列
    - 转为 numeric 后丢弃 NaN
    - 仅保留评分 1/3/5
    """
    tmp = df.copy()
    tmp = normalize_feature_naming(tmp)
    if "score_class" not in tmp.columns:
        if "score_label" not in tmp.columns:
            raise ValueError("DataFrame must contain score_label or score_class")
        tmp["score_class"] = tmp["score_label"].map(score_to_class_label)
    for col in feat_names:
        if col not in tmp.columns:
            return pd.DataFrame(columns=list(feat_names) + ["score_label", "score_class"])
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp["score_label"] = pd.to_numeric(tmp["score_label"], errors="coerce")
    tmp = tmp.dropna(subset=list(feat_names) + ["score_label", "score_class"])
    tmp = tmp[tmp["score_class"].isin(CLASS_ORDER)]
    return tmp


def _cov_ellipse_2d(x, y, n_std=2.0):
    """
    计算 2D 协方差椭圆参数（用于可视化分布包络，不代表显著性边界）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None
    try:
        cov = np.cov(x, y)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if np.any(eigvals <= 0) or not np.all(np.isfinite(eigvals)):
            return None
        width, height = 2.0 * n_std * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return float(np.mean(x)), float(np.mean(y)), float(width), float(height), float(angle)
    except Exception:
        return None


def _mask_inside_cov_ellipse_2d(x, y, n_std=2.0):
    """
    基于二维协方差椭圆返回“位于椭圆内部”的点掩码。
    当样本不足或协方差不可逆时，回退为保留全部点。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    inside_mask = np.zeros_like(valid_mask, dtype=bool)
    if np.count_nonzero(valid_mask) < 3:
        inside_mask[valid_mask] = True
        return inside_mask

    xv = x[valid_mask]
    yv = y[valid_mask]
    pts = np.column_stack([xv, yv])
    center = np.mean(pts, axis=0)
    try:
        cov = np.cov(pts.T)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            inside_mask[valid_mask] = True
            return inside_mask
        cov = cov + np.eye(2) * 1e-9
        inv_cov = np.linalg.inv(cov)
        deltas = pts - center
        md2 = np.einsum("ni,ij,nj->n", deltas, inv_cov, deltas)
        inside_mask[valid_mask] = md2 <= float(n_std ** 2)
        if not np.any(inside_mask[valid_mask]):
            inside_mask[valid_mask] = True
    except Exception:
        inside_mask[valid_mask] = True
    return inside_mask


def _trim_extreme_points_iqr_2d(x, y, whisker=2.2):
    """
    用 IQR 围栏先裁掉 2D 图中的极端离群点，避免少量异常值把坐标范围和椭圆严重拉大。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    xv = x[valid_mask]
    yv = y[valid_mask]
    if xv.size < 6:
        return xv, yv

    def _iqr_bounds(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 1e-12:
            return None
        return q1 - whisker * iqr, q3 + whisker * iqr

    x_bounds = _iqr_bounds(xv)
    y_bounds = _iqr_bounds(yv)
    if x_bounds is None and y_bounds is None:
        return xv, yv

    keep_mask = np.ones_like(xv, dtype=bool)
    if x_bounds is not None:
        keep_mask &= (xv >= x_bounds[0]) & (xv <= x_bounds[1])
    if y_bounds is not None:
        keep_mask &= (yv >= y_bounds[0]) & (yv <= y_bounds[1])

    if np.count_nonzero(keep_mask) < max(4, int(np.ceil(0.5 * xv.size))):
        return xv, yv
    return xv[keep_mask], yv[keep_mask]


def _filter_xy_by_cov_ellipse_2d(x, y, n_std=2.0):
    """
    返回用于绘图的 2D 点：
    - 能构建协方差椭圆时，仅保留椭圆内部点；
    - 不能构建椭圆时，保留全部有效点。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_valid, y_valid = _trim_extreme_points_iqr_2d(x, y)
    if x_valid.size == 0:
        return x_valid, y_valid, False

    can_use_ellipse = _cov_ellipse_2d(x_valid, y_valid, n_std=n_std) is not None
    if not can_use_ellipse:
        return x_valid, y_valid, False

    inside_mask = _mask_inside_cov_ellipse_2d(x_valid, y_valid, n_std=n_std)
    x_plot = x_valid[inside_mask]
    y_plot = y_valid[inside_mask]
    if x_plot.size == 0:
        return x_valid, y_valid, False
    return x_plot, y_plot, True


def _collect_display_coords_2d(coords, scores, n_std=2.0):
    """
    按类别汇总 2D 图实际会显示的点，用于计算更贴合图面的坐标范围。
    """
    coords = np.asarray(coords, dtype=float)
    scores = np.asarray(scores, dtype=float)
    class_labels = [score_to_class_label(s) for s in scores]
    xs = []
    ys = []
    all_used_ellipse = True

    for cls in CLASS_ORDER:
        mask = np.asarray([c == cls for c in class_labels])
        if not np.any(mask):
            continue
        x_plot, y_plot, used_ellipse = _filter_xy_by_cov_ellipse_2d(
            coords[mask, 0],
            coords[mask, 1],
            n_std=n_std,
        )
        if x_plot.size == 0:
            continue
        xs.append(x_plot)
        ys.append(y_plot)
        all_used_ellipse = all_used_ellipse and used_ellipse

    if not xs:
        return None, None, False
    return np.concatenate(xs), np.concatenate(ys), all_used_ellipse


def _add_cov_ellipse_2d(ax, x, y, color, n_std=2.0):
    if np.asarray(x).size < 3:
        print("[!] 警告：某类别 2D 样本少于 3，跳过协方差椭圆。")
        return
    params = _cov_ellipse_2d(x, y, n_std=n_std)
    if params is None:
        return
    cx, cy, width, height, angle = params
    ellipse = Ellipse(
        xy=(cx, cy),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.16,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(ellipse)


def _kde_1d(values, grid):
    """
    一维 KDE，失败时返回 None，避免影响整体绘图。
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3 or gaussian_kde is None:
        return None
    try:
        kde = gaussian_kde(vals)
        y = kde(grid)
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _feature_display_name(col):
    return _pretty_feature_label(col)


def _robust_long_tail_ylim(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    q90 = np.percentile(vals, 90)
    q95 = np.percentile(vals, 95)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        return _compute_axis_limits(vals)

    upper = max(q90 * 1.10, min(q95, q3 + 3.0 * iqr))
    upper = max(upper, q3 + 1.0 * iqr)
    upper = float(np.ceil((upper * 1.08) / 5.0) * 5.0)
    upper = min(upper, 30.0)
    lower = min(0.0, float(np.percentile(vals, 1)))
    return lower, upper


def _pretty_feature_label(name):
    mapping = {
        "H1H2_output": "H1H2 (dB)",
        "H1H2": "H1H2 (dB)",
        "CPP": "CPP (dB)",
        "Q1": "Q1 (dimensionless)",
        "QValue": "Q1 (dimensionless)",
        "HNR": "HNR (dB)",
        "SpectralSlope": "Spectral slope",
        "LowFreqEnergyRatio": "Low-frequency energy ratio",
        "HighFreqNoiseRatio": "HF residual-noise ratio",
        "Jitter": "Jitter",
        "Shimmer": "Shimmer",
    }
    return mapping.get(name, name)


def _plot_joint_2d_panel(
        fig,
        subspec,
        df_panel,
        x_col,
        y_col,
        panel_title,
        is_last=False,
        shared_xlim=None,
        shared_ylim=None,
        clip_to_limits=False,
        show_x_label=True,
        show_y_label=True,
        show_legend=True,
):
    """
    单个 2D joint panel：
    - 主图：scatter + 2D covariance ellipse
    - 上方：x histogram + KDE
    - 右侧：y histogram + KDE
    """
    inner = subspec.subgridspec(
        nrows=2, ncols=2,
        height_ratios=[0.9, 4.4],
        width_ratios=[4.4, 0.9],
        hspace=0.03, wspace=0.03
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
    ax_empty = fig.add_subplot(inner[0, 1])
    ax_empty.axis("off")

    clipped_total = 0
    for cls in CLASS_ORDER:
        grp = df_panel[df_panel["score_class"] == cls]
        if grp.empty:
            continue
        color = CLASS_COLOR_MAP[cls]
        if clip_to_limits:
            x = pd.to_numeric(grp[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp[y_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            used_ellipse = x.size >= 3
        else:
            x, y, used_ellipse = _filter_xy_by_cov_ellipse_2d(
                grp[x_col].values,
                grp[y_col].values,
                n_std=2.0,
            )
        if x.size == 0:
            continue

        if clip_to_limits and shared_ylim is not None:
            ymin, ymax = shared_ylim
            xmask = np.ones_like(x, dtype=bool)
            if shared_xlim is not None:
                xmask &= (x >= shared_xlim[0]) & (x <= shared_xlim[1])
            in_view = xmask & (y >= ymin) & (y <= ymax)
            above_view = xmask & (y > ymax)
            below_view = xmask & (y < ymin)
            clipped_total += int(np.count_nonzero(above_view) + np.count_nonzero(below_view))

            x_scatter = x[in_view]
            y_scatter = y[in_view]
            if x_scatter.size == 0 and not np.any(above_view) and not np.any(below_view):
                continue
        else:
            in_view = np.ones_like(x, dtype=bool)
            above_view = np.zeros_like(x, dtype=bool)
            below_view = np.zeros_like(x, dtype=bool)
            x_scatter = x
            y_scatter = y

        if x_scatter.size > 0:
            ax_main.scatter(
                x_scatter, y_scatter,
                s=20,
                alpha=0.78,
                c=color,
                edgecolors="#4a4a4a",
                linewidths=0.35,
                zorder=3,
            )
        if clip_to_limits and shared_ylim is not None:
            ymin, ymax = shared_ylim
            span = max(ymax - ymin, 1e-6)
            if np.any(above_view):
                ax_main.scatter(
                    x[above_view],
                    np.full(np.count_nonzero(above_view), ymax - 0.025 * span),
                    s=28,
                    marker="^",
                    alpha=0.85,
                    c=color,
                    edgecolors="#4a4a4a",
                    linewidths=0.35,
                    zorder=4,
                )
            if np.any(below_view):
                ax_main.scatter(
                    x[below_view],
                    np.full(np.count_nonzero(below_view), ymin + 0.025 * span),
                    s=28,
                    marker="v",
                    alpha=0.85,
                    c=color,
                    edgecolors="#4a4a4a",
                    linewidths=0.35,
                    zorder=4,
                )
        if used_ellipse and x_scatter.size >= 3:
            _add_cov_ellipse_2d(ax_main, x_scatter, y_scatter, color=color, n_std=2.0)

        if x_scatter.size > 0:
            ax_top.hist(x_scatter, bins=16, density=True, color=color, alpha=0.25, edgecolor="none")
            if shared_xlim is not None:
                x_grid = np.linspace(shared_xlim[0], shared_xlim[1], 220)
            else:
                x_grid = np.linspace(np.min(x_scatter), np.max(x_scatter), 220)
            x_kde = _kde_1d(x_scatter, x_grid)
            if x_kde is not None:
                ax_top.plot(x_grid, x_kde, color=color, linestyle="--", linewidth=1.4)

            ax_right.hist(y_scatter, bins=16, density=True, orientation="horizontal", color=color, alpha=0.25, edgecolor="none")
            if shared_ylim is not None:
                y_grid = np.linspace(shared_ylim[0], shared_ylim[1], 220)
            else:
                y_grid = np.linspace(np.min(y_scatter), np.max(y_scatter), 220)
            y_kde = _kde_1d(y_scatter, y_grid)
            if y_kde is not None:
                ax_right.plot(y_kde, y_grid, color=color, linestyle="--", linewidth=1.4)

    if shared_xlim is not None:
        ax_main.set_xlim(*shared_xlim)
    if shared_ylim is not None:
        ax_main.set_ylim(*shared_ylim)

    ax_top.set_title(panel_title, fontsize=11, pad=3)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_main.grid(True, linestyle="--", alpha=0.26)
    ax_main.set_xlabel(_feature_display_name(x_col) if show_x_label else "", fontsize=10)
    ax_main.set_ylabel(_feature_display_name(y_col) if show_y_label else "", fontsize=10)
    if not show_y_label:
        ax_main.tick_params(axis="y", labelleft=False)
    if clipped_total:
        ax_main.text(
            0.012,
            0.965,
            f"{clipped_total} Q1 outlier(s) shown at axis limit",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="w", markerfacecolor=CLASS_COLOR_MAP[cls],
               markeredgecolor="#4a4a4a", markeredgewidth=0.35, markersize=7, label=cls)
        for cls in CLASS_ORDER
    ]
    if show_legend:
        ax_main.legend(handles=legend_handles, title=None, fontsize=8, loc="upper right", frameon=True)
    if (not is_last) or (not show_x_label):
        ax_main.tick_params(axis="x", labelbottom=False)


def generate_paper_2d_joint_horizontal(
        df_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="CPP",
        pitch_name="chest",
):
    """
    论文风格 2D joint 可视化（横向 Pressed phonation/Breathy phonation/ALL）：
    仅展示分布结构（散点 + 协方差椭圆 + 边际分布），不是分类边界。
    """
    df_stats = normalize_feature_naming(df_stats)
    df_full = remove_outlier_jitter_df(enrich_df_with_metadata(df_stats))

    panel_dfs = []
    for subset_group in SUBSET_GROUPS:
        df_subset = filter_df_by_subset(df_full, subset_group)
        panel_dfs.append(_clean_numeric_panel_df(df_subset, [x_col, y_col]))

    valid_panels = [d for d in panel_dfs if not d.empty]
    if not valid_panels:
        print(f"[!] 无可用数据，跳过 paper 2D 图：{x_col} vs {y_col}")
        return

    x_all = np.concatenate([d[x_col].values for d in valid_panels])
    y_all = np.concatenate([d[y_col].values for d in valid_panels])
    shared_xlim = _compute_axis_limits(x_all)
    use_robust_y_zoom = y_col in {"Q1", "QValue"}
    shared_ylim = _robust_long_tail_ylim(y_all) if use_robust_y_zoom else _compute_axis_limits(y_all)

    fig = plt.figure(figsize=(15.0, 4.9), dpi=PAPER_DPI, facecolor="white")
    outer = fig.add_gridspec(nrows=1, ncols=3, wspace=0.16)
    for i, (panel_df, panel_title) in enumerate(zip(panel_dfs, SUBSET_TITLES)):
        if panel_df.empty:
            ax = fig.add_subplot(outer[0, i])
            ax.axis("off")
            ax.set_title(f"{panel_title} (No Data)", fontsize=11)
            continue
        _plot_joint_2d_panel(
            fig, outer[0, i], panel_df, x_col, y_col, panel_title,
            is_last=True, shared_xlim=shared_xlim, shared_ylim=shared_ylim,
            clip_to_limits=use_robust_y_zoom,
            show_x_label=True,
            show_y_label=(i == 0),
            show_legend=(i == 2),
        )

    fig.suptitle(
        f"{pitch_name} - {_pretty_feature_label(x_col)} vs {_pretty_feature_label(y_col)}",
        fontsize=13,
        y=0.995,
    )
    out_dir = os.path.join(outputs_root, "plot_2d_joint_horizontal")
    stem = f"{pitch_name}_{_feature_file_token(x_col)}_vs_{_feature_file_token(y_col)}_horizontal"
    saved = _save_figure(fig, out_dir, stem, dpi=PAPER_DPI, save_png=True, save_jpg=False)
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, dpi=PAPER_DPI, bbox_inches="tight")
    if use_robust_y_zoom:
        zoom_stem = f"{stem}_zoomed"
        saved.extend(_save_figure(fig, out_dir, zoom_stem, dpi=PAPER_DPI, save_png=True, save_jpg=False))
        fig.savefig(os.path.join(out_dir, f"{zoom_stem}.pdf"), dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)
    for p in saved:
        print(f"[+] Paper 2D 图已保存：{p}")
    print(f"[+] Paper 2D 图已保存：{pdf_path}")


def generate_paper_2d_joint_vertical(*args, **kwargs):
    """Backward-compatible wrapper: paper 2D triptychs are now generated horizontally."""
    return generate_paper_2d_joint_horizontal(*args, **kwargs)


def _ellipsoid_mesh_from_cov(points, n_std=2.0, n_u=32, n_v=16):
    """
    基于 3D 协方差构建椭球网格，用于可视化分布包络，不代表统计显著边界。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        return None
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 4:
        return None
    try:
        mean = np.mean(pts, axis=0)
        cov = np.cov(pts.T)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
            return None
        cov = cov + np.eye(3) * 1e-9
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        if not np.all(np.isfinite(eigvals)):
            return None

        u = np.linspace(0, 2 * np.pi, n_u)
        v = np.linspace(0, np.pi, n_v)
        uu, vv = np.meshgrid(u, v)
        sphere = np.stack([
            np.cos(uu) * np.sin(vv),
            np.sin(uu) * np.sin(vv),
            np.cos(vv),
        ], axis=0)  # (3, n_v, n_u)

        radii = n_std * np.sqrt(eigvals)
        transform = eigvecs @ np.diag(radii)
        flat = sphere.reshape(3, -1)
        ellip = (mean[:, None] + transform @ flat).reshape(3, *uu.shape)
        return ellip[0], ellip[1], ellip[2]
    except Exception:
        return None


def _plot_3d_panel(
        ax,
        df_panel,
        x_col,
        y_col,
        z_col,
        panel_title,
        shared_lims=None,
        show_legend=True,
):
    for cls in CLASS_ORDER:
        grp = df_panel[df_panel["score_class"] == cls]
        if grp.empty:
            continue
        color = CLASS_COLOR_MAP[cls]
        pts = grp[[x_col, y_col, z_col]].values
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=24,
            alpha=0.85,
            c=color,
            edgecolors="k",
            linewidths=0.3,
            label=cls,
        )
        mesh = _ellipsoid_mesh_from_cov(pts, n_std=2.0, n_u=28, n_v=14)
        if mesh is not None:
            X, Y, Z = mesh
            ax.plot_surface(
                X, Y, Z,
                color=color,
                alpha=0.13,
                linewidth=0,
                shade=False,
                antialiased=True,
            )
        else:
            if pts.shape[0] < 4:
                print("[!] 警告：某类别 3D 样本少于 4，跳过协方差椭球。")

    ax.set_xlabel(_feature_display_name(x_col), fontsize=9, labelpad=2)
    ax.set_ylabel(_feature_display_name(y_col), fontsize=9, labelpad=2)
    ax.set_zlabel(_feature_display_name(z_col), fontsize=9, labelpad=3)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_title(panel_title, fontsize=11, pad=1)
    ax.view_init(elev=22, azim=-60)
    ax.grid(True, linestyle="--", alpha=0.25)

    if shared_lims is not None:
        xlim, ylim, zlim = shared_lims
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color="w",
                   markerfacecolor=CLASS_COLOR_MAP[cls],
                   markeredgecolor="k", markeredgewidth=0.3,
                   markersize=7, label=cls)
            for cls in CLASS_ORDER
        ]
        ax.legend(handles=handles, title=None, loc="upper right", fontsize=8, frameon=True)


def _autocrop_saved_image_whitespace(
        image_path,
        white_threshold=248,
        margin_px=14,
):
    """
    自动裁白边：仅裁掉近白背景区域，并保留安全边距，避免误裁标题/标签/图例。
    """
    if Image is None:
        return image_path
    if not os.path.isfile(image_path):
        return image_path
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        return image_path
    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
        # 非近白像素视为有效内容
        content_mask = np.any(arr < white_threshold, axis=2)
        coords = np.argwhere(content_mask)
        if coords.size == 0:
            return image_path

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        h, w = arr.shape[:2]
        x0 = max(0, int(x0) - margin_px)
        y0 = max(0, int(y0) - margin_px)
        x1 = min(w, int(x1) + margin_px)
        y1 = min(h, int(y1) + margin_px)

        # 避免极小裁剪带来的抖动（小于 6px 不裁）
        if x0 <= 6 and y0 <= 6 and (w - x1) <= 6 and (h - y1) <= 6:
            return image_path

        with Image.open(image_path) as im:
            cropped = im.crop((x0, y0, x1, y1))
            cropped.save(image_path)
    except Exception:
        return image_path
    return image_path


def generate_paper_3d_ellipsoid_vertical(
        df_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="Q1",
        z_col="CPP",
        pitch_name="chest",
):
    """
    论文风格 3D 可视化（竖向 Pressed phonation/Breathy phonation/ALL）：
    3D scatter + covariance ellipsoid，用于展示分布趋势而非分类边界。
    """
    df_stats = normalize_feature_naming(df_stats)
    df_full = remove_outlier_jitter_df(enrich_df_with_metadata(df_stats))

    panel_dfs = []
    for subset_group in SUBSET_GROUPS:
        df_subset = filter_df_by_subset(df_full, subset_group)
        panel_dfs.append(_clean_numeric_panel_df(df_subset, [x_col, y_col, z_col]))

    valid_panels = [d for d in panel_dfs if not d.empty]
    if not valid_panels:
        print(f"[!] 无可用数据，跳过 paper 3D 图：{x_col} vs {y_col} vs {z_col}")
        return

    x_all = np.concatenate([d[x_col].values for d in valid_panels])
    y_all = np.concatenate([d[y_col].values for d in valid_panels])
    z_all = np.concatenate([d[z_col].values for d in valid_panels])
    shared_lims = (
        _compute_axis_limits(x_all),
        _compute_axis_limits(y_all),
        _compute_axis_limits(z_all),
    )

    fig = plt.figure(figsize=(8.4, 15.0), dpi=PAPER_DPI, facecolor="white")
    axes = [
        fig.add_subplot(3, 1, 1, projection="3d"),
        fig.add_subplot(3, 1, 2, projection="3d"),
        fig.add_subplot(3, 1, 3, projection="3d"),
    ]
    for i, (ax, panel_df, panel_title) in enumerate(zip(axes, panel_dfs, SUBSET_TITLES)):
        if panel_df.empty:
            ax.set_axis_off()
            continue
        _plot_3d_panel(
            ax=ax,
            df_panel=panel_df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            panel_title=panel_title,
            shared_lims=shared_lims,
            show_legend=True if i == 0 else False,
        )

    fig.suptitle(
        f"{pitch_name} - {_pretty_feature_label(x_col)} vs {_pretty_feature_label(y_col)} vs {_pretty_feature_label(z_col)}",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.975, bottom=0.03, hspace=0.30)
    out_dir = os.path.join(outputs_root, "plot_3d_ellipsoid_vertical")
    stem = (
        f"{pitch_name}_{_feature_file_token(x_col)}_vs_"
        f"{_feature_file_token(y_col)}_vs_{_feature_file_token(z_col)}_vertical"
    )
    saved = _save_figure(
        fig, out_dir, stem, dpi=PAPER_DPI, save_png=True, save_jpg=False, bbox_tight=False
    )
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, dpi=PAPER_DPI, bbox_inches=None)
    plt.close(fig)
    for p in saved:
        print(f"[+] Paper 3D 图已保存：{p}")
    print(f"[+] Paper 3D 图已保存：{pdf_path}")


def generate_paper_3d_ellipsoid_horizontal(
        df_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="Q1",
        z_col="CPP",
        pitch_name="chest",
):
    """
    论文风格 3D 可视化（横向 Pressed phonation/Breathy phonation/ALL）：
    3D scatter + covariance ellipsoid，用于展示分布趋势而非分类边界。
    """
    df_stats = normalize_feature_naming(df_stats)
    df_full = remove_outlier_jitter_df(enrich_df_with_metadata(df_stats))

    panel_dfs = []
    for subset_group in SUBSET_GROUPS:
        df_subset = filter_df_by_subset(df_full, subset_group)
        panel_dfs.append(_clean_numeric_panel_df(df_subset, [x_col, y_col, z_col]))

    valid_panels = [d for d in panel_dfs if not d.empty]
    if not valid_panels:
        print(f"[!] 无可用数据，跳过 paper 3D 横版图：{x_col} vs {y_col} vs {z_col}")
        return

    x_all = np.concatenate([d[x_col].values for d in valid_panels])
    y_all = np.concatenate([d[y_col].values for d in valid_panels])
    z_all = np.concatenate([d[z_col].values for d in valid_panels])
    shared_lims = (
        _compute_axis_limits(x_all),
        _compute_axis_limits(y_all),
        _compute_axis_limits(z_all),
    )

    # 横向论文图：略微拉长宽度，同时保持紧凑高度
    fig = plt.figure(figsize=(14.1, 4.3), dpi=PAPER_DPI, facecolor="white")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    for i, (ax, panel_df, panel_title) in enumerate(zip(axes, panel_dfs, SUBSET_TITLES)):
        if panel_df.empty:
            ax.set_axis_off()
            continue
        _plot_3d_panel(
            ax=ax,
            df_panel=panel_df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            panel_title=panel_title,
            shared_lims=shared_lims,
            show_legend=True if i == 0 else False,
        )

    fig.suptitle(
        f"{pitch_name} - {_pretty_feature_label(x_col)} vs {_pretty_feature_label(y_col)} vs {_pretty_feature_label(z_col)}",
        fontsize=13,
        y=0.999,
    )
    # 先整体压缩边距，再手动精调 3D 轴位置（tight_layout 对 3D 控制不稳定）
    fig.subplots_adjust(left=0.015, right=0.992, top=0.885, bottom=0.01, wspace=0.11)
    for ax in axes:
        pos = ax.get_position()
        new_y0 = max(0.0, pos.y0 - 0.01)
        # 顶部限制在 0.90，给 suptitle 与三个面板标题留出清晰间隔，避免视觉重复/重叠
        new_h = min(0.90 - new_y0, pos.height + 0.02)
        ax.set_position([pos.x0, new_y0, pos.width, new_h])
    out_dir = os.path.join(outputs_root, "plot_3d_ellipsoid_horizontal")
    stem = (
        f"{pitch_name}_{_feature_file_token(x_col)}_vs_"
        f"{_feature_file_token(y_col)}_vs_{_feature_file_token(z_col)}_horizontal"
    )
    saved = _save_figure(
        fig,
        out_dir,
        stem,
        dpi=PAPER_DPI,
        save_png=True,
        save_jpg=False,
        bbox_tight=True,
        pad_inches=0.012,
    )
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)

    # matplotlib 3D 常见残留白边的兜底处理
    for p in saved:
        _autocrop_saved_image_whitespace(p, white_threshold=248, margin_px=8)

    for p in saved:
        print(f"[+] Paper 3D 横版图已保存：{p}")
    print(f"[+] Paper 3D 横版图已保存：{pdf_path}")


def plot_scatter_ndim(
        df_stats: pd.DataFrame,
        outputs_root: str,
        ndim: int,
        feat_names: list = None
):
    """
    绘制散点图的统一入口函数 (DataFrame 版本)，根据 ndim 参数决定绘制 1D/2D/3D 散点图。

    在 outputs_root 下的 "plot_1d" / "plot_2d" / "plot_3d" 目录中保存所有生成的图像。
    根据 ndim 参数，排列组合出所有 ndim 个特征统计值的组合，
    为每个特征统计值组合创建一个图像。
    每个图像分三个子图，分别展示 pressed phonation、breathy phonation 和 ALL 子集的特征值与打分标签的关系。
    - 1D: x 轴为评分标签，y 轴为一个特征的统计量，点的颜色和形状根据音高区分。
    - 2D/3D: x, y(, z) 轴为声学特征统计值，点的颜色代表评分标签。

    Args:
        df_stats: 特征统计 DataFrame (索引为文件名，列为特征值)。
        outputs_root: 输出目录路径。
        ndim: 维度 (1, 2, 或 3)。
    """
    if ndim < 1 or ndim > 3:
        raise ValueError("ndim 必须为 1, 2 或 3")

    # 1. 数据预处理
    df_stats = normalize_feature_naming(df_stats)
    df_full = enrich_df_with_metadata(df_stats)  # 添加元数据列
    df_full = remove_outlier_jitter_df(df_full)  # 移除 jitter 异常值
    # 检查是否有有效数据
    if df_full.empty:
        print("[!] 数据为空，无法绘图。")
        return

    # 提取声学特征列名 (排除元数据列)
    meta_cols = ['score_label', 'score_class', 'subset_type', 'pitch_digit']
    available_feat_names = [c for c in df_full.columns if c not in meta_cols]
    if feat_names:
        all_feat_names = [f for f in feat_names if f in available_feat_names]
        missing_feats = [f for f in feat_names if f not in available_feat_names]
        if missing_feats:
            print(f"[!] 以下配置特征在统计表中不存在，将跳过绘图：{missing_feats}")
    else:
        all_feat_names = available_feat_names
    if not all_feat_names:
        print("[!] 未找到任何特征列。")
        return
    # 根据 ndim 确定特征组合
    if ndim == 1:
        feat_combinations = [(f,) for f in all_feat_names]
    else:
        feat_combinations = list(combinations(all_feat_names, ndim))
    # 创建输出目录
    plot_dir = os.path.join(outputs_root, f"plot_{ndim}d")
    os.makedirs(plot_dir, exist_ok=True)

    # 2. 遍历特征组合进行绘图
    all_1d_summary_rows = []
    for feat_tuple in feat_combinations:
        # 1D：按 Pressed phonation/Breathy phonation/ALL 三个子集横向拼接
        if ndim == 1:
            feat = feat_tuple[0]
            if feat not in df_full.columns:
                continue
            subset_payloads = []
            for subset_group in SUBSET_GROUPS:
                df_subset = filter_df_by_subset(df_full, subset_group)
                if feat in df_subset.columns:
                    df_subset[feat] = pd.to_numeric(df_subset[feat], errors="coerce")
                df_subset["score_label"] = pd.to_numeric(df_subset["score_label"], errors="coerce")
                df_valid = df_subset.dropna(subset=[feat, "score_label"])
                df_valid = df_valid[df_valid["score_label"].isin(sorted(SCORE_TO_CLASS.keys()))]
                if df_valid.empty:
                    subset_payloads.append(None)
                    continue
                subset_payloads.append({
                    "coords": df_valid[[feat]].values,
                    "scores": df_valid["score_label"].values.astype(float),
                    "df": df_valid.copy(),
                })

            valid_payloads = [p for p in subset_payloads if p is not None]
            if not valid_payloads:
                print(f"[!] 跳过：{feat} 在三个子集中均无有效数据。")
                continue

            y_all = np.concatenate([p["coords"][:, 0] for p in valid_payloads])
            shared_ylim = _compute_axis_limits(y_all)
            feature_summary_rows = []
            for panel_title, payload in zip(SUBSET_TITLES, subset_payloads):
                if payload is None:
                    continue
                rows = _summarize_1d_panel(payload["df"], feat, panel_title)
                feature_summary_rows.extend(rows)
                all_1d_summary_rows.extend(rows)
            if feature_summary_rows:
                summary_df = pd.DataFrame(feature_summary_rows)
                summary_path = os.path.join(
                    plot_dir,
                    f"summary_{_feature_file_token(feat)}_by_panel_category.csv",
                )
                summary_df.to_csv(summary_path, index=False)

            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), dpi=DPI, sharex=False, sharey=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

            for i, _subset_group in enumerate(SUBSET_GROUPS):
                ax = axes[i]
                payload = subset_payloads[i]
                if payload is None:
                    ax.set_axis_off()
                    ax.set_title(f"{SUBSET_TITLES[i]} (No Data)")
                    continue

                _plot_1d_scatter(
                    ax,
                    payload["coords"],
                    payload["scores"],
                    None,
                    feat_tuple,
                    is_last_subset=(i == 2),
                )
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
                ax.set_title(SUBSET_TITLES[i], fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.3)
                if i > 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

            fig.suptitle(f"{_pretty_feature_label(feat)}", fontsize=16)
            fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.91])

            stem = f"chest_{_feature_file_token(feat)}_1d"
            _save_figure(fig, plot_dir, stem, dpi=300)
            plt.close(fig)
            continue

        # 2D / 3D：按 Pressed phonation/Breathy phonation/ALL 三个子集横向拼接
        subset_payloads = []
        for subset_group in SUBSET_GROUPS:
            df_subset = filter_df_by_subset(df_full, subset_group)
            need_cols = list(feat_tuple) + ["score_label"]
            for c in feat_tuple:
                if c in df_subset.columns:
                    df_subset[c] = pd.to_numeric(df_subset[c], errors="coerce")
            df_subset["score_label"] = pd.to_numeric(df_subset["score_label"], errors="coerce")
            df_valid = df_subset.dropna(subset=need_cols)
            df_valid = df_valid[df_valid["score_label"].isin(sorted(SCORE_TO_CLASS.keys()))]
            if df_valid.empty:
                subset_payloads.append(None)
                continue

            subset_payloads.append({
                "coords": df_valid[list(feat_tuple)].values,
                "scores": df_valid['score_label'].values.astype(float),
            })

        valid_payloads = [p for p in subset_payloads if p is not None]
        if not valid_payloads:
            print(f"[!] 跳过：{feat_tuple} 在三个子集中均无有效数据。")
            continue

        shared_xlim = None
        shared_ylim = None
        shared_zlim = None
        if ndim == 2:
            display_xs = []
            display_ys = []
            all_used_ellipse = True
            for payload in valid_payloads:
                x_disp, y_disp, used_ellipse = _collect_display_coords_2d(
                    payload["coords"],
                    payload["scores"],
                    n_std=2.0,
                )
                if x_disp is None or y_disp is None:
                    continue
                display_xs.append(x_disp)
                display_ys.append(y_disp)
                all_used_ellipse = all_used_ellipse and used_ellipse
            if display_xs and display_ys:
                pad_ratio = 0.035 if all_used_ellipse else 0.07
                shared_xlim = _compute_axis_limits(np.concatenate(display_xs), pad_ratio=pad_ratio)
                shared_ylim = _compute_axis_limits(np.concatenate(display_ys), pad_ratio=pad_ratio)
        elif ndim == 3:
            x_all = np.concatenate([p["coords"][:, 0] for p in valid_payloads])
            y_all = np.concatenate([p["coords"][:, 1] for p in valid_payloads])
            z_all = np.concatenate([p["coords"][:, 2] for p in valid_payloads])
            shared_xlim = _compute_axis_limits(x_all)
            shared_ylim = _compute_axis_limits(y_all)
            shared_zlim = _compute_axis_limits(z_all)

        if ndim == 3:
            fig = plt.figure(figsize=(14.1, 4.3), dpi=DPI)
            axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

            for i, _subset_group in enumerate(SUBSET_GROUPS):
                ax = axes[i]
                payload = subset_payloads[i]
                if payload is None:
                    ax.set_axis_off()
                    ax.set_title(f"{SUBSET_TITLES[i]} (No Data)")
                    continue

                coords = payload["coords"]
                scores = payload["scores"]
                _plot_3d_scatter(ax, coords, scores, feat_tuple, show_legend=(i == 2))

                if shared_xlim is not None:
                    ax.set_xlim(*shared_xlim)
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
                if shared_zlim is not None:
                    ax.set_zlim(*shared_zlim)

                ax.set_title(SUBSET_TITLES[i], fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3)
        else:
            fig = plt.figure(figsize=(15.0, 4.9), dpi=DPI)
            outer = fig.add_gridspec(1, 3, wspace=0.16)

            for i, _subset_group in enumerate(SUBSET_GROUPS):
                payload = subset_payloads[i]
                if payload is None:
                    ax = fig.add_subplot(outer[0, i])
                    ax.set_axis_off()
                    ax.set_title(f"{SUBSET_TITLES[i]} (No Data)")
                    continue

                coords = payload["coords"]
                scores = payload["scores"]
                df_panel = pd.DataFrame(
                    {
                        feat_tuple[0]: coords[:, 0],
                        feat_tuple[1]: coords[:, 1],
                        "score_label": scores,
                    }
                )
                df_panel["score_class"] = [
                    score_to_class_label(s) for s in df_panel["score_label"].values
                ]

                _plot_joint_2d_panel(
                    fig,
                    outer[0, i],
                    df_panel,
                    feat_tuple[0],
                    feat_tuple[1],
                    panel_title=SUBSET_TITLES[i],
                    is_last=True,
                    shared_xlim=shared_xlim,
                    shared_ylim=shared_ylim,
                    show_x_label=True,
                    show_y_label=(i == 0),
                    show_legend=(i == 2),
                )

        if ndim == 2:
            fig.suptitle(
                f"{_pretty_feature_label(feat_tuple[0])} vs {_pretty_feature_label(feat_tuple[1])}",
                fontsize=16,
                y=0.995,
            )
            stem = f"chest_{_feature_file_token(feat_tuple[0])}_vs_{_feature_file_token(feat_tuple[1])}_2d_horizontal"
        else:
            fig.suptitle(
                f"{_pretty_feature_label(feat_tuple[0])} vs {_pretty_feature_label(feat_tuple[1])} vs {_pretty_feature_label(feat_tuple[2])}",
                fontsize=13,
                y=0.995,
            )
            stem = (
                f"chest_{_feature_file_token(feat_tuple[0])}_vs_"
                f"{_feature_file_token(feat_tuple[1])}_vs_{_feature_file_token(feat_tuple[2])}_3d_horizontal"
            )
        if ndim == 2:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout",
                    category=UserWarning,
                )
                fig.tight_layout(rect=[0.012, 0.01, 0.995, 0.93], pad=0.24)
        else:
            fig.subplots_adjust(left=0.015, right=0.992, top=0.865, bottom=0.02, wspace=0.11)
        _save_figure(fig, plot_dir, stem, dpi=300, bbox_tight=True, pad_inches=0.015)
        plt.close(fig)

    if ndim == 1 and all_1d_summary_rows:
        pd.DataFrame(all_1d_summary_rows).to_csv(
            os.path.join(plot_dir, "summary_all_1d_features_by_panel_category.csv"),
            index=False,
        )

    print(f"[+] {ndim}D 散点图绘制完成，共生成 {len(feat_combinations)} 张图像。")


if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    dataset_name = cfg.dataset.name
    score_file = cfg.dataset.score_file
    acoustic_feats = cfg.acoustic_feats

    outputs_root = os.path.join(proj_root, "outputs")
    raw_feats_dir = os.path.join(outputs_root, "raw_feats", dataset_name)
    if not os.path.isdir(raw_feats_dir):
        alt_raw_feats_dir = os.path.join(outputs_root, dataset_name, "raw_feats")
        if os.path.isdir(alt_raw_feats_dir):
            raw_feats_dir = alt_raw_feats_dir
            print(f"[*] 使用数据集分层输出目录：{raw_feats_dir}")
    print("[*] 开始提取特征统计信息...")
    from src.feat_extract.feat_extractor import extract_feats_stats_from_csv
    df_feats_stats = extract_feats_stats_from_csv(raw_feats_dir)
    print("[*] 开始绘制散点图...")
    for ndim in [1, 2, 3]:
        print(f"[*] 绘制 {ndim}D 散点图...")
        plot_scatter_ndim(df_feats_stats, outputs_root, ndim, feat_names=list(acoustic_feats))

    generate_paper_2d_joint_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="CPP",
        pitch_name="chest",
    )
    generate_paper_2d_joint_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="Q1",
        pitch_name="chest",
    )
    generate_paper_2d_joint_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="HNR",
        pitch_name="chest",
    )
    generate_paper_3d_ellipsoid_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="Q1",
        z_col="CPP",
        pitch_name="chest",
    )
    generate_paper_3d_ellipsoid_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="H1H2_output",
        y_col="HNR",
        z_col="CPP",
        pitch_name="chest",
    )
    generate_paper_3d_ellipsoid_horizontal(
        df_feats_stats,
        outputs_root,
        x_col="HNR",
        y_col="Q1",
        z_col="CPP",
        pitch_name="chest",
    )
    print("[+] 散点图绘制完成！")
