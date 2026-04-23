import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

from src.data_parser import parse_label, parse_suffix_type, parse_pitch_digit


# ==============================================================================
# 配置与常量
# ==============================================================================

# 定义数据集子集组合，分别为 A+1、B+1、A+B+1（All）
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
MARKER_SIZE = 80
# 样式映射常量
PITCH_MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
SCORE_COLOR_MAP = {1: 'red', 3: 'green', 5: 'blue'}


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


# ==============================================================================
# 绘图工具函数
# ==============================================================================

def get_pitch_style_map(unique_pitches):
    """生成一致的音高颜色和形状映射"""
    color_map = {p: plt.cm.tab10(i) for i, p in enumerate(unique_pitches)}
    marker_map = {
        p: marker
        for p, marker in zip(unique_pitches, PITCH_MARKERS[:len(unique_pitches)])
    }
    return color_map, marker_map


def create_legend_handles_pitch(color_map, marker_map, ax):
    """创建音高图例句柄"""
    handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_map[p],
            color='w',
            label=f'Pitch {p}',
            markerfacecolor=color_map[p],
            markersize=8,
        )
        for p in sorted(color_map.keys())
    ]
    return handles


def create_legend_handles_score(ax):
    """创建评分图例句柄"""
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o', color='w',
            label=f'Score {score}',
            markerfacecolor=SCORE_COLOR_MAP[score],
            markersize=8,
        )
        for score in sorted(SCORE_COLOR_MAP.keys())
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


def _plot_1d_scatter(ax, coords, scores, pitches_arr, feat_tuple, is_last_subset):
    """
    绘制 1D 散点图逻辑。x 轴为评分标签 (带抖动)，y 轴为特征值。点的颜色和形状根据音高区分。
    """
    x_coords = scores + np.random.uniform(
        -PLOT_JITTER_RANGE, PLOT_JITTER_RANGE, size=scores.shape
    )
    y_coords = coords[:, 0]

    # 绘制散点，根据音高区分点的样式
    unique_p = sorted(list(set([p for p in pitches_arr if p is not None])))
    if not unique_p:
        # 如果没有音高信息，全部用一种样式
        ax.scatter(x_coords, y_coords, c='gray', s=MARKER_SIZE, alpha=ALPHA)
    else:
        c_map, m_map = get_pitch_style_map(unique_p)
        for p_val in unique_p:
            mask = pitches_arr == p_val
            if np.any(mask):
                ax.scatter(
                    x_coords[mask], y_coords[mask],
                    c=[c_map[p_val]], marker=m_map[p_val],
                    s=MARKER_SIZE, alpha=ALPHA,
                )

    ax.set_xlabel("Score Label")
    ax.set_ylabel(f"{feat_tuple[0]} Value")
    ax.set_xticks([1, 3, 5])

    # 只在最后一个子图添加图例
    if is_last_subset and unique_p:
        handles = create_legend_handles_pitch(c_map, m_map, ax)
        ax.legend(handles=handles, title="Pitch", loc='upper right', fontsize=8)


def _plot_2d_scatter(ax, coords, scores, feat_tuple, is_last_subset):
    """
    绘制 2D 散点图逻辑。x, y 轴为两个特征值，点的颜色代表评分标签。
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    colors = [SCORE_COLOR_MAP.get(int(s), 'gray') for s in scores]
    ax.scatter(x_coords, y_coords, c=colors, s=MARKER_SIZE, alpha=ALPHA)
    ax.set_xlabel(feat_tuple[0])
    ax.set_ylabel(feat_tuple[1])
    # 只在最后一个子图添加图例
    if is_last_subset:
        handles = create_legend_handles_score(ax)
        ax.legend(handles=handles, title="Score Label", loc='upper right', fontsize=8)


def _plot_3d_scatter(ax, coords, scores, feat_tuple):
    """
    绘制 3D 散点图逻辑。x, y, z 轴为三个特征值。点的颜色代表评分标签。
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    colors = [SCORE_COLOR_MAP.get(int(s), 'gray') for s in scores]
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=MARKER_SIZE, alpha=ALPHA)
    ax.set_xlabel(feat_tuple[0])
    ax.set_ylabel(feat_tuple[1])
    ax.set_zlabel(feat_tuple[2])
    handles = create_legend_handles_score(ax)
    ax.legend(handles=handles, title="Score", loc='upper left', fontsize=8)


def plot_scatter_ndim(df_stats: pd.DataFrame, outputs_root: str, ndim: int):
    """
    绘制散点图的统一入口函数 (DataFrame 版本)，根据 ndim 参数决定绘制 1D/2D/3D 散点图。

    在 outputs_root 下的 "plot_1d" / "plot_2d" / "plot_3d" 目录中保存所有生成的图像。
    根据 ndim 参数，排列组合出所有 ndim 个特征统计值的组合，
    为每个特征统计值组合创建一个图像。
    每个图像分三个子图，分别展示不同数据子集（A+1、B+1、A+B+1（All））的特征值与打分标签的关系。
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
    df_full = enrich_df_with_metadata(df_stats)  # 添加元数据列
    df_full = remove_outlier_jitter_df(df_full)  # 移除 jitter 异常值
    # 检查是否有有效数据
    if df_full.empty:
        print("[!] 数据为空，无法绘图。")
        return

    # 提取所有声学特征列名 (排除元数据列)
    meta_cols = ['score_label', 'subset_type', 'pitch_digit']
    all_feat_names = [c for c in df_full.columns if c not in meta_cols]
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
    for feat_tuple in feat_combinations:
        # 先准备三个子集的数据，后续统一计算坐标轴范围
        subset_payloads = []
        for subset_group in SUBSET_GROUPS:
            df_subset = filter_df_by_subset(df_full, subset_group)
            df_valid = df_subset.dropna(subset=['score_label'])
            df_valid = df_valid.dropna(subset=['pitch_digit'])
            if df_valid.empty:
                subset_payloads.append(None)
                continue

            subset_payloads.append({
                "coords": df_valid[list(feat_tuple)].values,
                "scores": df_valid['score_label'].values.astype(float),
                "pitches_arr": df_valid['pitch_digit'].values,
            })

        valid_payloads = [p for p in subset_payloads if p is not None]
        shared_xlim = None
        shared_ylim = None
        shared_zlim = None
        if valid_payloads:
            if ndim == 1:
                y_all = np.concatenate([p["coords"][:, 0] for p in valid_payloads])
                shared_ylim = _compute_axis_limits(y_all)
            elif ndim == 2:
                x_all = np.concatenate([p["coords"][:, 0] for p in valid_payloads])
                y_all = np.concatenate([p["coords"][:, 1] for p in valid_payloads])
                shared_xlim = _compute_axis_limits(x_all)
                shared_ylim = _compute_axis_limits(y_all)
            elif ndim == 3:
                x_all = np.concatenate([p["coords"][:, 0] for p in valid_payloads])
                y_all = np.concatenate([p["coords"][:, 1] for p in valid_payloads])
                z_all = np.concatenate([p["coords"][:, 2] for p in valid_payloads])
                shared_xlim = _compute_axis_limits(x_all)
                shared_ylim = _compute_axis_limits(y_all)
                shared_zlim = _compute_axis_limits(z_all)

        # 设置子图
        if ndim == 3:
            fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
            axes = [fig.add_subplot(1, 3, i + 1, projection='3d')
                    for i in range(3)]
        else:
            # 1D 和 2D 使用普通子图，1D 共享 Y 轴
            sharey = (ndim == 1)
            fig, axes_temp = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI, sharey=sharey)
            if not isinstance(axes_temp, (list, np.ndarray)):
                axes = [axes_temp]
            else:
                axes = axes_temp

        # 遍历每个子集组 (A+1, B+1, All)
        for i, subset_group in enumerate(SUBSET_GROUPS):
            ax = axes[i]
            payload = subset_payloads[i]
            if payload is None:
                ax.set_axis_off()
                ax.set_title(f"Subset: {'+'.join(subset_group)} (No Data)")
                continue

            # 提取坐标数据
            coords = payload["coords"]
            scores = payload["scores"]
            pitches_arr = payload["pitches_arr"]

            # --- 调用对应的维度绘图逻辑 ---
            if ndim == 1:
                _plot_1d_scatter(
                    ax, coords, scores, pitches_arr, feat_tuple,
                    is_last_subset=(i == 2)
                )
            elif ndim == 2:
                _plot_2d_scatter(
                    ax, coords, scores, feat_tuple,
                    is_last_subset=(i == 2)
                )
            elif ndim == 3:
                _plot_3d_scatter(
                    ax, coords, scores, feat_tuple
                )

            # 强制对齐三个子图的坐标轴范围
            if ndim == 1:
                ax.set_xlim(0.5, 5.5)
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
            elif ndim == 2:
                if shared_xlim is not None:
                    ax.set_xlim(*shared_xlim)
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
            elif ndim == 3:
                if shared_xlim is not None:
                    ax.set_xlim(*shared_xlim)
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
                if shared_zlim is not None:
                    ax.set_zlim(*shared_zlim)

            ax.set_title(f"Subset: {'+'.join(subset_group)}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

        # 设置总标题
        title_suffix = f"{feat_tuple[0]}" if ndim == 1 else f"{', '.join(feat_tuple)}"
        fig.suptitle(f"{ndim}D Scatter Plot: {title_suffix}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图像
        filename_safe = "_".join(feat_tuple)
        plot_path = os.path.join(plot_dir, f"scatter_{ndim}d_{filename_safe}.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        # print(f"[+] 已保存图像：{plot_path}")  # 可选：减少控制台输出

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
    print("[*] 开始提取特征统计信息...")
    from src.feat_extract.feat_extractor import extract_feats_stats_from_csv
    df_feats_stats = extract_feats_stats_from_csv(raw_feats_dir)
    print("[*] 开始绘制散点图...")
    for ndim in [1, 2, 3]:
        print(f"[*] 绘制 {ndim}D 散点图...")
        plot_scatter_ndim(df_feats_stats, outputs_root, ndim)
    print("[+] 散点图绘制完成！")
