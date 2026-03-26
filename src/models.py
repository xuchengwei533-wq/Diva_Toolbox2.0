import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.combined_data import CombinedData


def _prepare_xy(
        dataset: CombinedData,
        tech_name: str,
        subsets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """从 CombinedData 对象中提取特定技巧和子集的 X, y。"""
    df_subset = dataset.get_one_score_feats_subset(tech_name, subsets)
    y = df_subset[tech_name].values.astype(np.float32)
    X = df_subset[dataset.feat_cols].values.astype(np.float32)
    return df_subset, X, y


def run_pca_analysis(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """执行 PCA 分析。"""
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行 PCA | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return

    # 标准化 & PCA
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n_comp = min(len(dataset.feat_cols), Xz.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(Xz)

    # 结果保存
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 保存原始数据
    df_subset.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)

    # 1. Loadings CSV
    comps = pd.DataFrame(
        pca.components_,
        columns=dataset.feat_cols,
        index=[f"PC{i + 1}" for i in range(n_comp)]
    )
    comps.to_csv(os.path.join(out_path, "pca_loadings.csv"))

    # 2. Variance CSV
    var_df = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(n_comp)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        },
    )
    var_df.to_csv(os.path.join(out_path, "pca_explained_variance.csv"), index=False)

    # 3. PC1 Formula
    pc1 = comps.loc["PC1"].sort_values(key=abs, ascending=False)
    formula = " + ".join([f"{v:.4f}*{k}" for k, v in pc1.items()])
    with open(os.path.join(out_path, "pc1_formula.txt"), "w", encoding="utf-8") as f:
        f.write(formula)

    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(comps.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dataset.feat_cols)))
    ax.set_xticklabels(dataset.feat_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(n_comp)], fontsize=9)
    for i in range(n_comp):
        for j in range(len(dataset.feat_cols)):
            val = comps.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if abs(val) > 0.6 else "black", fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"PCA Loadings ({tech_name} - {subset_label})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "pca_heatmap.png"), dpi=300)
    plt.close(fig)

    print(f"[+] PCA 完成 (解释方差 Top2: {pca.explained_variance_ratio_[:2].sum():.2%})")


def run_lasso_analysis(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """执行 LASSO 分析。"""
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行 LASSO | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return

    # 标准化 & LASSO
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=10000)
    model.fit(Xz, y)

    # 结果整理
    coefs = pd.Series(model.coef_, index=dataset.feat_cols)
    coef_df = pd.DataFrame(
        {
            "feature": dataset.feat_cols,
            "coef": coefs.values,
            "abs_coef": np.abs(coefs.values),
        },
    ).sort_values(by="abs_coef", ascending=False)
    coef_df["selected"] = np.abs(coef_df["coef"]) >= 0.01

    # 保存
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 0. 保存原始数据
    df_subset.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)
    # 1. 保存系数 CSV
    coef_df.to_csv(os.path.join(out_path, "lasso_coefficients.csv"), index=False)
    # 2. 绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    # 准备数据
    features = coef_df["feature"].tolist()
    coefficients = coef_df["coef"].values
    n_features = len(features)
    # 生成颜色列表
    colors = ['#1f77b4' if c > 0 else '#d62728' for c in coefficients]
    ax.bar(features, coefficients, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_ylabel("Coefficient (Standardized)")
    ax.set_title(f"LASSO Coefficients ({tech_name} - {subset_label})")
    ax.set_ylim(-1, 1)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "lasso_barplot.png"), dpi=300)
    plt.close(fig)

    selected = coef_df[coef_df["selected"]]["feature"].tolist()
    print(f"[+] LASSO 完成 (选中特征: {len(selected)})")


def run_ordinal_regression(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """
    执行序数回归分析 (Ordinal Regression)。
    使用 statsmodels OrderedModel (Probit/Logit link) 分析特征对有序评分的影响。
    """
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行序数回归 | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return
    unique_scores = sorted(np.unique(y))

    # 标准化特征 (X)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    feature_cols = dataset.feat_cols
    # 构建 DataFrame 以便 statsmodels 使用
    df_model = pd.DataFrame(Xz, columns=feature_cols)
    df_model['target'] = y.astype(int)

    # 构建模型
    y_cat = pd.Categorical(df_model['target'], categories=unique_scores, ordered=True)
    try:
        model = OrderedModel(endog=y_cat, exog=df_model[feature_cols], distr='probit')
        result = model.fit(method='bfgs', maxiter=10000, disp=False)
    except Exception as e:
        print(f"[!] 模型拟合失败: {e}")
        return

    # 保存结果
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 0. 保存原始数据集 (含评分和原始特征)
    df_output = df_subset.copy()
    df_output.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)
    # 1. 保存标准化后的数据集 (用于建模的数据)
    df_scaled = df_model.copy()
    df_scaled.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score_scaled.csv"), index=False)
    # 2. 保存回归系数表
    # 提取统计量，整理结果表格
    mask = [idx in feature_cols for idx in result.params.index]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": result.params[mask].values,
            "std_err": result.bse[mask].values,
            "z_value": result.tvalues[mask].values,
            "p_value": result.pvalues[mask].values,
            "odds_ratio": np.exp(result.params[mask].values),
            "or_ci_lower": np.exp(result.conf_int()[mask][0]),
            "or_ci_upper": np.exp(result.conf_int()[mask][1]),
        }
    )
    # 按系数绝对值排序 (影响度)
    coef_df["impact"] = np.abs(coef_df["coef"])
    coef_df = coef_df.sort_values(by="impact", ascending=False).reset_index(drop=True)
    coef_df.to_csv(os.path.join(out_path, "ordinal_regression_coefficients.csv"), index=False)
    # 3. 保存模型摘要文本
    summary_text = result.summary().as_text()
    with open(os.path.join(out_path, "ordinal_regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Ordinal Regression Summary\n")
        f.write(f"Technique: {tech_name}\n")
        f.write(f"Subset: {subset_label}\n")
        f.write(f"Link Function: Probit\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_text)

    significant_feats = coef_df[coef_df["p_value"] < 0.05]["feature"].tolist()
    print(f"[+] 序数回归完成 | 显著特征 (p<0.05): {len(significant_feats)}")
    if significant_feats:
        print(f"    -> {', '.join(significant_feats[:5])}{'...' if len(significant_feats) > 5 else ''}")


def run_correlation_matrix(
        dataset: CombinedData,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """
    构建 [技巧 x 特征] 的全局关联性矩阵。
    使用 Spearman 秩相关系数，适用于有序评分与连续特征。
    """
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行全局关联矩阵分析 | 子集：{subset_label}")

    tech_cols = dataset.tech_cols  # 10个技巧
    feat_cols = dataset.feat_cols  # 9个声学特征

    # 存储结果的列表
    results_long = []

    # 用于构建矩阵的字典 {tech: {feat: corr}}
    matrix_data = {tech: {} for tech in tech_cols}
    matrix_pvals = {tech: {} for tech in tech_cols}
    matrix_slopes = {tech: {} for tech in tech_cols}

    # 获取全量数据 (合并所有技巧列)
    df_all = dataset.get_scores_feats_subset(subset_types)
    if df_all.empty:
        print("[!] 错误：无法提取有效数据。")
        return
    print(f"[*] 总样本量：{len(df_all)}")

    # 遍历每一个技巧 (Y) 和 每一个特征 (X)
    for tech in tech_cols:
        y = df_all[tech].dropna().values
        # 获取当前技巧有效的索引，确保 X 和 Y 对齐
        valid_idx = df_all[tech].notna()

        for feat in feat_cols:
            x = df_all.loc[valid_idx, feat].values
            if len(x) != len(y):
                continue
            x_arr = x.astype(np.float64)
            y_arr = y.astype(np.float64)

            # 1. 计算 Spearman 相关系数
            corr, p_val = spearmanr(x_arr, y_arr)

            # 2. 计算简单线性回归斜率 (仅用于指示方向和相对强度，非因果)
            # 先标准化 X 以便斜率具有可比性 (表示 X 变动 1 个标准差，Y 变动多少)
            if np.std(x_arr) == 0:
                slope = 0.0
            else:
                z_x = (x_arr - np.mean(x_arr)) / np.std(x_arr)
                slope, _, _, _, _ = linregress(z_x, y_arr)

            # 记录结果
            results_long.append(
                {
                    "technique": tech,
                    "feature": feat,
                    "spearman_corr": corr,
                    "p_value": p_val,
                    "slope_std": slope,  # 标准化后的斜率
                    "sample_size": len(x),
                    "significant": p_val < 0.05
                }
            )

            # 填充矩阵数据
            matrix_data[tech][feat] = corr
            matrix_pvals[tech][feat] = p_val
            matrix_slopes[tech][feat] = slope

    # --- 结果整理与保存 ---
    out_path = os.path.join(output_dir, "global_correlation_matrix")
    os.makedirs(out_path, exist_ok=True)

    # 1. 保存长表 (详细数据)
    df_long = pd.DataFrame(results_long)
    df_long.to_csv(os.path.join(out_path, "correlation_details.csv"), index=False)
    print(f"[+] 已保存详细数据：correlation_details.csv")

    # 2. 构建并保存相关性矩阵 (Heatmap Data)
    df_corr_matrix = pd.DataFrame(matrix_data).T  # 行：技巧，列：特征
    df_corr_matrix = df_corr_matrix[feat_cols]  # 确保列顺序一致
    df_corr_matrix.to_csv(os.path.join(out_path, "correlation_matrix_spearman.csv"))

    # 3. 构建并保存 P值矩阵 (用于标记显著性)
    df_pval_matrix = pd.DataFrame(matrix_pvals).T
    df_pval_matrix = df_pval_matrix[feat_cols]
    df_pval_matrix.to_csv(os.path.join(out_path, "correlation_matrix_pvalues.csv"))

    # 4. 绘制热力图
    plt.figure(figsize=(12, 8), dpi=150)
    # 使用掩膜隐藏不显著的结果 (可选，这里我们显示所有但用星号标记)
    # 这里直接画相关系数
    sns.heatmap(
        df_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Spearman Correlation Coefficient"}
    )
    plt.title(f"Global Correlation Matrix: Techniques vs Acoustic Features\n(Subsets: {subset_label})", fontsize=14)
    plt.xlabel("Acoustic Features", fontsize=12)
    plt.ylabel("Vocal Techniques", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print(f"[+] 已保存热力图：correlation_heatmap.png")

    # 5. 打印显著关联摘要
    sig_df = df_long[df_long["significant"]].sort_values(by="spearman_corr", key=abs, ascending=False)
    print(f"\n[+] 发现 {len(sig_df)} 个显著关联 (p<0.05):")
    if not sig_df.empty:
        print(sig_df[["technique", "feature", "spearman_corr", "p_value"]].to_string(index=False))
    else:
        print("    (无显著关联)")


def run_lasso_correlation_matrix(
        dataset: CombinedData,
        output_dir: str,
        subset_types: list = None,
):
    """
    构建 [技巧 x 特征] 的 Lasso 关联矩阵。
    矩阵值：标准化后的回归系数。0 表示该特征被 Lasso 剔除（无独立关联）。
    """
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行 Lasso 关联矩阵 | 子集：{subset_label}")

    tech_cols = dataset.tech_cols
    feat_cols = dataset.feat_cols

    # 获取全量数据
    df_all = dataset.get_scores_feats_subset(subset_types)
    if df_all.empty:
        return
    print(f"[*] 总样本量：{len(df_all)}")

    # 初始化矩阵
    lasso_matrix = pd.DataFrame(np.nan, index=tech_cols, columns=feat_cols)
    lasso_selection_count = pd.Series(0, index=feat_cols)  # 统计每个特征被选中的次数

    for tech in tech_cols:
        # 准备数据
        valid_idx = df_all[tech].notna()
        y = df_all.loc[valid_idx, tech].values
        X = df_all.loc[valid_idx, feat_cols].values

        # 标准化
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        # 运行 LassoCV
        try:
            model = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=100)
            model.fit(Xz, y)
            coefs = model.coef_
        except Exception as e:
            print(f"[!] Lasso 拟合失败 ({tech}): {e}")
            coefs = np.zeros(len(feat_cols))  # 失败则视为全0

        # 填入矩阵
        for i, feat in enumerate(feat_cols):
            val = coefs[i]
            # 设定一个极小的阈值视为 0 (数值噪声)
            if abs(val) < 1e-6:
                val = 0.0
            lasso_matrix.loc[tech, feat] = val
            if val != 0.0:
                lasso_selection_count[feat] += 1

    # --- 保存与绘图 ---
    os.makedirs(output_dir, exist_ok=True)
    lasso_matrix.to_csv(os.path.join(output_dir, "lasso_correlation_matrix.csv"))
    # 绘图
    plt.figure(figsize=(12, 8), dpi=150)
    # 使用 diverging colormap, 0 为白色/中性
    sns.heatmap(
        lasso_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,  # 可根据实际数据调整范围
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Standardized Coefficient"}
    )
    plt.title(f"Lasso Correlation Matrix (Multivariate)\n(Subsets: {subset_label})", fontsize=14)
    plt.xlabel("Acoustic Features")
    plt.ylabel("Vocal Techniques")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lasso_heatmap.png"), dpi=300)
    plt.close()

    print(f"[+] Lasso 矩阵完成。保存于: {output_dir}")
    print(f"    特征被选中频次:\n{lasso_selection_count.sort_values(ascending=False)}")


def run_ordinal_correlation_matrix(
        dataset: CombinedData,
        output_dir: str,
        subset_types: list = None,
        metric: str = "coef"  # 可选: "coef" (系数) 或 "or" (优势比)
):
    """
    构建 [技巧 x 特征] 的序数回归关联矩阵。
    矩阵值：
      - 如果 metric='coef': 标准化系数 (正负表示方向)。
      - 如果 metric='or': 优势比 (大于1为正相关，小于1为负相关)。
    仅当 p < 0.05 且模型收敛时填入，否则为 NaN。
    """
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行序数回归关联矩阵 ({metric}) | 子集：{subset_label}")

    tech_cols = dataset.tech_cols
    feat_cols = dataset.feat_cols

    df_all = dataset.get_scores_feats_subset(subset_types)
    if df_all.empty:
        return

    # 初始化矩阵
    ord_matrix = pd.DataFrame(np.nan, index=tech_cols, columns=feat_cols)
    ord_pval_matrix = pd.DataFrame(np.nan, index=tech_cols, columns=feat_cols)
    convergence_stats = {"success": 0, "failed": 0, "separation": 0}

    for tech in tech_cols:
        valid_idx = df_all[tech].notna()
        y = df_all.loc[valid_idx, tech].values.astype(int)
        X = df_all.loc[valid_idx, feat_cols].values

        # 标准化
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        df_model = pd.DataFrame(Xz, columns=feat_cols)

        y_cat = pd.Categorical(y, categories=sorted(np.unique(y)), ordered=True)
        try:
            # 拟合模型
            model = OrderedModel(endog=y_cat, exog=df_model, distr='probit')
            result = model.fit(method='bfgs', maxiter=5000, disp=False)

            # 【关键检查】1. 收敛性
            if not result.mle_retvals.get('converged', False):
                convergence_stats["failed"] += 1
                continue  # 不收敛则不填入

            # 【关键检查】2. 系数合理性 (防止完全分离导致的爆炸)
            params = result.params
            if np.any(np.abs(params) > 10):  # 阈值设为 10，超过视为异常
                convergence_stats["separation"] += 1
                # print(f"[!] 检测到分离现象 ({tech})，跳过该行。")
                continue

            # 提取结果
            pvals = result.pvalues
            coeffs = params

            for i, feat in enumerate(feat_cols):
                p = pvals.iloc[i] if hasattr(pvals, 'iloc') else pvals[i]
                c = coeffs.iloc[i] if hasattr(coeffs, 'iloc') else coeffs[i]

                if p < 0.05:  # 仅保留显著项
                    if metric == "coef":
                        ord_matrix.loc[tech, feat] = c
                    elif metric == "or":
                        ord_matrix.loc[tech, feat] = np.exp(c)

                    ord_pval_matrix.loc[tech, feat] = p

            convergence_stats["success"] += 1
        except Exception as e:
            convergence_stats["failed"] += 1
            # print(f"[!] 序数回归报错 ({tech}): {e}")

    # --- 保存与绘图 ---
    os.makedirs(output_dir, exist_ok=True)
    ord_matrix.to_csv(os.path.join(output_dir, f"ordinal_correlation_matrix_{metric}.csv"))
    ord_pval_matrix.to_csv(os.path.join(output_dir, "ordinal_pvalues_matrix.csv"))

    # 绘图配置
    if metric == "or":
        # OR 值绘图需要特殊处理，因为 1 是中点，且刻度是对数的
        # 为了热力图美观，通常对 OR 取 log，或者使用特殊的 cmap
        # 这里我们直接画 log(OR)，这样 0 是中点，正负分明
        plot_data = np.log(ord_matrix)
        center_val = 0
        cmap_name = "coolwarm"
        cbar_label = "Log(Odds Ratio)"
        fmt_val = ".2f"
        # 注意：annot 显示的是 log 值，如果需要显示原始 OR 值，需要自定义 annot
        # 为了简单，这里显示 log 值，或者我们可以手动格式化 annot
        annot_data = ord_matrix.round(2)  # 显示原始 OR 值在格子里
    else:
        plot_data = ord_matrix
        center_val = 0
        cmap_name = "RdBu_r"
        cbar_label = "Coefficient (Probit)"
        fmt_val = ".2f"
        annot_data = ord_matrix.round(2)
    plt.figure(figsize=(12, 8), dpi=150)

    # 使用 mask 隐藏 NaN
    mask = ord_matrix.isnull()
    ax = sns.heatmap(
        plot_data,
        annot=annot_data,
        fmt=fmt_val,
        cmap=cmap_name,
        center=center_val,
        mask=mask,  # 隐藏不显著/失败的区域
        linewidths=0.5,
        cbar_kws={"label": cbar_label}
    )

    title_suffix = "Log(Odds Ratio)" if metric == "or" else "Coefficients"
    plt.title(f"Ordinal Regression Matrix ({title_suffix}, p<0.05)\n(Subsets: {subset_label})", fontsize=14)
    plt.xlabel("Acoustic Features")
    plt.ylabel("Vocal Techniques")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ordinal_heatmap_{metric}.png"), dpi=300)
    plt.close()

    print(f"[+] 序数回归矩阵 ({metric}) 完成。")
    print(
        f"    统计: 成功={convergence_stats['success']}, "
        f"未收敛={convergence_stats['failed']}, "
        f"分离/异常={convergence_stats['separation']}"
    )


if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    data_root = cfg.dataset.root_dir
    dataset_name = cfg.dataset.name
    score_file = cfg.dataset.score_file
    subset_groups = cfg.dataset.subset_groups
    print(f"[*] 数据集：{dataset_name} | 评分文件：{score_file} | 子集分组：{subset_groups}")
    outputs_root = os.path.join(proj_root, "outputs")
    raw_feats_dir = os.path.join(outputs_root, "raw_feats", dataset_name)
    analysis_dir = os.path.join(outputs_root, "analysis")
    matrix_dir = os.path.join(outputs_root, "matrix")

    print("[*] 开始提取特征统计信息...")
    from src.feat_extract.feat_extractor import extract_feats_stats_from_csv
    df_stats = extract_feats_stats_from_csv(raw_feats_dir)
    print(df_stats.head())
    print(f"[*] 加载评分矩阵：{score_file}")
    from src.data_loader import load_score_matrix
    score_path = os.path.join(data_root, dataset_name, score_file)
    df_score = load_score_matrix(score_path)
    print(df_score.head())
    print("[*] 开始合并评分矩阵和特征统计信息...")
    from src.combined_data import CombinedData
    combined_data = CombinedData(df_score, df_stats)
    print("[+] 合并完成！")

    print("[*] 开始分析...")
    # for tech in combined_data.tech_cols:
    #     for subset_group in subset_groups:
    #         run_pca_analysis(combined_data, tech, analysis_dir, subset_group)
    #         run_lasso_analysis(combined_data, tech, analysis_dir, subset_group)
    #         run_ordinal_regression(combined_data, tech, analysis_dir, subset_group)

    run_correlation_matrix(combined_data, matrix_dir, ["B", "1"])
    run_lasso_correlation_matrix(combined_data, matrix_dir, ["B", "1"])
    run_ordinal_correlation_matrix(combined_data, matrix_dir, ["B", "1"], metric="coef")
    run_ordinal_correlation_matrix(combined_data, matrix_dir, ["B", "1"], metric="or")
