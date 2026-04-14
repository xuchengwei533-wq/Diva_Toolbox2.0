import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.combined_data import CombinedData


def _prepare_xy_safe(
        dataset: CombinedData,
        tech_name: str,
        subsets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """从 CombinedData 对象中提取特定技巧和子集的 X, y。"""
    df_subset = dataset.get_one_score_feats_subset(tech_name, subsets)
    df_subset = df_subset.dropna(subset=[tech_name] + dataset.feat_cols)
    if len(df_subset) == 0:
        return pd.DataFrame(), np.array([]), np.array([])
    y = df_subset[tech_name].values.astype(int)  # 确保是整数
    X = df_subset[dataset.feat_cols].values.astype(float)
    return df_subset, X, y


def run_ordinal_regression_with_cv(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subsets: Optional[List[str]] = None,
):
    """
    执行序数回归，并使用留一法 (LOOCV) 进行严格验证。
    输出：系数表 + LOOCV 准确率/Kappa 值。
    """
    subset_label = "+".join(subsets) if subsets else "All"
    print(f"[*] 运行增强序数回归 | 技巧：{tech_name} | 子集：{subset_label}")

    df_subset, X, y = _prepare_xy_safe(dataset, tech_name, subsets)
    if len(y) < 10:
        print("[!] 跳过：样本量太少 (<10)。")
        return

    n_samples = len(y)
    unique_scores = sorted(np.unique(y))

    # 标准化
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    feature_cols = dataset.feat_cols

    # --- 步骤 1: 全数据拟合 (获取公式系数) ---
    df_model = pd.DataFrame(Xz, columns=feature_cols)
    y_cat = pd.Categorical(y, categories=unique_scores, ordered=True)

    try:
        model_full = OrderedModel(endog=y_cat, exog=df_model, distr='probit')
        result_full = model_full.fit(method='bfgs', maxiter=5000, disp=False)

        if not result_full.mle_retvals.get('converged', False):
            print("[!] 警告：全数据模型未收敛。")

    except Exception as e:
        print(f"[!] 模型拟合失败：{e}")
        return

    # 提取系数
    mask = [idx in feature_cols for idx in result_full.params.index]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": result_full.params[mask].values,
            "p_value": result_full.pvalues[mask].values,
            "significant": result_full.pvalues[mask].values < 0.05
        }
    )

    # --- 步骤 2: 留一法交叉验证 (LOOCV) ---
    # 小样本必须用 LOOCV，不能用 K-Fold (K=5 会浪费太多训练数据)
    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    print(f"    -> 正在进行 LOOCV (n={n_samples})...")

    for train_idx, test_idx in loo.split(Xz):
        X_train, X_test = Xz[train_idx], Xz[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        df_train = pd.DataFrame(X_train, columns=feature_cols)
        y_train_cat = pd.Categorical(y_train, categories=unique_scores, ordered=True)

        try:
            # 重新拟合
            model_loo = OrderedModel(endog=y_train_cat, exog=df_train, distr='probit')
            res_loo = model_loo.fit(method='bfgs', maxiter=2000, disp=False)

            if not res_loo.mle_retvals.get('converged', False):
                # 如果单次不收敛，预测为众数
                pred_class = np.bincount(y_train).argmax()
            else:
                # 预测概率并取最大概率类
                # statsmodels 的 predict 返回的是累积概率或类别概率，需查看文档
                # 这里使用 predict 方法获取类别
                pred_probs = res_loo.predict(exog=pd.DataFrame(X_test, columns=feature_cols))
                # pred_probs shape: (1, num_categories)
                pred_class = unique_scores[np.argmax(pred_probs.values[0])]

        except Exception:
            pred_class = np.bincount(y_train).argmax()  # 失败则 fallback

        y_true_loo.append(y_test[0])
        y_pred_loo.append(pred_class)

    # 计算指标
    acc = accuracy_score(y_true_loo, y_pred_loo)
    kappa = cohen_kappa_score(y_true_loo, y_pred_loo, weights='quadratic')
    mae = mean_absolute_error(y_true_loo, y_pred_loo)  # 对于有序变量，MAE 也有意义

    # --- 保存结果 ---
    out_path = os.path.join(output_dir, tech_name, subset_label, "ordinal_enhanced")
    os.makedirs(out_path, exist_ok=True)

    # 1. 系数表
    coef_df.to_csv(os.path.join(out_path, "coefficients.csv"), index=False)

    # 2. 验证报告
    report = {
        "metric": ["Accuracy", "Quadratic Kappa", "MAE"],
        "value": [acc, kappa, mae],
        "sample_size": [n_samples, n_samples, n_samples]
    }
    pd.DataFrame(report).to_csv(os.path.join(out_path, "cv_metrics.csv"), index=False)

    # 3. 生成公式文本 (Probit 链接函数下的线性组合)
    # 注意：序数回归的公式是 P(Y<=k) = Phi(threshold_k - sum(beta*x))
    # 这里我们写出线性部分 sum(beta*x)
    sig_coefs = coef_df[coef_df['significant']]
    formula_str = " + ".join([f"{row['coef']:.4f}*{row['feature']}" for _, row in sig_coefs.iterrows()])
    with open(os.path.join(out_path, "linear_formula.txt"), "w") as f:
        f.write(f"Linear Predictor (Z) = {formula_str}\n")
        f.write(f"P(Score <= k) = Probit_CDF(Threshold_k - Z)\n")
        f.write(f"\nLOOCV Accuracy: {acc:.3f}\n")
        f.write(f"LOOCV Kappa: {kappa:.3f}\n")

    print(f"[+] 序数回归完成 | Acc: {acc:.3f}, Kappa: {kappa:.3f}")
    print(f"    显著特征：{sig_coefs['feature'].tolist()}")


def run_ordinal_regression_with_stratified_kfold_cv(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subsets: Optional[List[str]] = None,
        n_splits: int = 10,
):
    """
    执行序数回归，并使用分层 K 折交叉验证（默认 10 折）评估泛化表现。
    会根据最小类别样本数自动下调折数，无法分层时自动跳过。
    """
    subset_label = "+".join(subsets) if subsets else "All"
    print(f"[*] 运行分层K折序数回归 | 技巧：{tech_name} | 子集：{subset_label}")

    df_subset, X, y = _prepare_xy_safe(dataset, tech_name, subsets)
    if len(y) < 10:
        print("[!] 跳过：样本量太少 (<10)。")
        return

    unique_scores = sorted(np.unique(y))
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    effective_splits = min(int(n_splits), min_class_count)
    if effective_splits < 2:
        print("[!] 跳过：最小类别样本数不足，无法执行分层交叉验证。")
        return

    feature_cols = dataset.feat_cols
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.transform(X_test)

        df_train = pd.DataFrame(X_train_z, columns=feature_cols)
        y_train_cat = pd.Categorical(y_train, categories=unique_scores, ordered=True)

        try:
            model = OrderedModel(endog=y_train_cat, exog=df_train, distr='probit')
            result = model.fit(method='bfgs', maxiter=3000, disp=False)

            if not result.mle_retvals.get('converged', False):
                pred = np.full(shape=len(y_test), fill_value=np.bincount(y_train).argmax())
            else:
                pred_probs = result.predict(exog=pd.DataFrame(X_test_z, columns=feature_cols))
                if hasattr(pred_probs, "values"):
                    pred_probs = pred_probs.values
                pred_idx = np.argmax(pred_probs, axis=1)
                pred = np.array([unique_scores[i] for i in pred_idx], dtype=int)
        except Exception:
            pred = np.full(shape=len(y_test), fill_value=np.bincount(y_train).argmax())

        fold_acc = accuracy_score(y_test, pred)
        fold_mae = mean_absolute_error(y_test, pred)
        try:
            fold_kappa = cohen_kappa_score(y_test, pred, weights='quadratic')
        except Exception:
            fold_kappa = np.nan

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": fold_acc,
                "quadratic_kappa": fold_kappa,
                "mae": fold_mae,
            }
        )
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_mae = mean_absolute_error(y_true_all, y_pred_all)
    try:
        overall_kappa = cohen_kappa_score(y_true_all, y_pred_all, weights='quadratic')
    except Exception:
        overall_kappa = np.nan

    out_path = os.path.join(output_dir, tech_name, subset_label, "ordinal_kfold")
    os.makedirs(out_path, exist_ok=True)

    pd.DataFrame(fold_rows).to_csv(os.path.join(out_path, "fold_metrics.csv"), index=False)
    pd.DataFrame(
        [
            {"metric": "Accuracy", "value": overall_acc},
            {"metric": "Quadratic Kappa", "value": overall_kappa},
            {"metric": "MAE", "value": overall_mae},
            {"metric": "n_splits_used", "value": effective_splits},
            {"metric": "sample_size", "value": len(y)},
        ]
    ).to_csv(os.path.join(out_path, "cv_metrics.csv"), index=False)
    pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(
        os.path.join(out_path, "cv_predictions.csv"), index=False
    )

    print(
        f"[+] 分层K折完成 | splits={effective_splits} | "
        f"Acc={overall_acc:.3f}, Kappa={overall_kappa:.3f}, MAE={overall_mae:.3f}"
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
    outputs_dir = os.path.join(outputs_root, "regression")
    target_techs = cfg.vocal_techs

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

    subsets = ["B", "1"]
    for target_tech in target_techs:
        # 运行增强序数回归
        run_ordinal_regression_with_cv(combined_data, target_tech, outputs_dir, subsets)
