import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

from src.data_parser import parse_label, parse_pitch_digit, parse_suffix_type
from src.plot_scatter import score_to_class_label


FEATURE_DISPLAY_NAMES = {
    "H1H2_output": "H1H2 (dB)",
    "CPP": "CPP (dB)",
    "Q1": "Q1 (dimensionless)",
    "HNR": "HNR (dB)",
    "SpectralSlope": "Spectral slope",
    "LowFreqEnergyRatio": "Low-frequency energy ratio",
    "Jitter": "Jitter",
    "Shimmer": "Shimmer",
}

TERM_DISPLAY_NAMES = {
    "C(score_class, Treatment(reference='A'))[T.B]": "State B vs A",
    "C(score_class, Treatment(reference='A'))[T.C]": "State C vs A",
    "pitch_c": "Pitch",
    "C(score_class, Treatment(reference='A'))[T.B]:pitch_c": "State B x Pitch",
    "C(score_class, Treatment(reference='A'))[T.C]:pitch_c": "State C x Pitch",
}

CORE_FEATURES = ["H1H2_output", "Q1", "CPP", "HNR"]
SEPARATION_FEATURES = ["H1H2_output", "Q1", "CPP", "HNR"]
LMM_FEATURES = [
    "H1H2_output",
    "Q1",
    "CPP",
    "HNR",
    "SpectralSlope",
    "LowFreqEnergyRatio",
    "Jitter",
    "Shimmer",
]
SUBSET_MAP = {
    "Pressed phonation": ["A", "1"],
    "Breathy phonation": ["B", "1"],
    "ALL": ["A", "B", "1"],
}


def _singer_id(filename: str) -> str:
    return str(filename).split("__")[0]


def _audio_stem(filename: str) -> str:
    return Path(str(filename)).stem


def _load_feature_data(outputs_root: Path) -> pd.DataFrame:
    df = pd.read_csv(outputs_root / "feats_data.csv", index_col=0)
    df.index.name = "audio_filename"
    df["score_label"] = [parse_label(idx) for idx in df.index]
    df["score_class"] = [score_to_class_label(v) for v in df["score_label"]]
    df["subset_type"] = [parse_suffix_type(idx) for idx in df.index]
    df["pitch_digit"] = [parse_pitch_digit(idx) for idx in df.index]
    df["singer_id"] = [_singer_id(idx) for idx in df.index]
    df["audio_stem"] = [_audio_stem(idx) for idx in df.index]
    metadata_cols = ["score_label", "score_class", "subset_type", "pitch_digit", "singer_id", "audio_stem"]
    analysis_cols = [col for col in LMM_FEATURES + metadata_cols if col in df.columns]
    return df[analysis_cols].copy()


def _remove_max_jitter(df: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    max_idx = df["Jitter"].idxmax()
    max_val = float(df.loc[max_idx, "Jitter"])
    return df.drop(index=max_idx).copy(), str(max_idx), max_val


def _assignable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["subset_type"].isin(["1", "A", "B"]) & df["score_class"].isin(["A", "B", "C"])].copy()


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def make_data_flow_tables(df_raw: pd.DataFrame, df_clean: pd.DataFrame, data_dir: Path, out_dir: Path, outlier_name: str, outlier_jitter: float):
    wav_count = len(list(data_dir.glob("*.wav"))) if data_dir.exists() else np.nan
    assignable = _assignable(df_clean)
    unassigned = df_clean[~df_clean.index.isin(assignable.index)]
    flow_rows = [
        {
            "stage": "WAV files visible in data/all-CUN",
            "n": wav_count,
            "note": "File-level count in the local data directory; may include recordings not summarized as feature tokens.",
        },
        {
            "stage": "Feature-summary tokens in feats_data.csv",
            "n": len(df_raw),
            "note": "Tokens with complete cached acoustic feature summaries.",
        },
        {
            "stage": "Removed maximum-Jitter outlier",
            "n": 1,
            "note": f"{outlier_name}; Jitter={outlier_jitter:.6f}.",
        },
        {
            "stage": "Tokens retained after outlier removal",
            "n": len(df_clean),
            "note": "Analysis-ready feature table before series assignment.",
        },
        {
            "stage": "Tokens assigned to Pressed/Breathy/ALL analysis panels",
            "n": len(assignable),
            "note": "Tokens with parseable suffix -1, -A, or -B and A/B/C score labels; this is the basis of Fig. 3-6 summaries.",
        },
        {
            "stage": "Tokens retained for model-level analyses",
            "n": len(df_clean),
            "note": "Includes tokens with valid acoustic features and A/B/C score labels; some do not enter series-specific figures.",
        },
        {
            "stage": "Tokens not assigned to series figures",
            "n": len(unassigned),
            "note": "Tokens without parseable -1/-A/-B suffix, retained for model-level sensitivity where appropriate.",
        },
    ]
    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_csv(out_dir / "data_flow_summary.csv", index=False)

    rows = []
    for panel, allowed in SUBSET_MAP.items():
        sub = df_clean[df_clean["subset_type"].isin(allowed) & df_clean["score_class"].isin(["A", "B", "C"])]
        for cat in ["A", "B", "C"]:
            rows.append(
                {
                    "analysis_panel": panel,
                    "category": cat,
                    "n": int((sub["score_class"] == cat).sum()),
                    "note": "Reference A tokens are reused in both Pressed and Breathy phonation panels." if panel in ["Pressed phonation", "Breathy phonation"] and cat == "A" else "",
                }
            )
    series_df = pd.DataFrame(rows)
    series_df.to_csv(out_dir / "series_category_counts.csv", index=False)

    missing = df_clean[df_clean["subset_type"].isna()].copy()
    missing[["score_label", "score_class", "singer_id", "pitch_digit"]].to_csv(out_dir / "unassigned_series_tokens.csv")
    return flow_df, series_df


def _fit_lasso(X: np.ndarray, y: np.ndarray, seed: int = 42):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    cv = max(2, min(5, len(y)))
    model = LassoCV(cv=cv, random_state=seed, alphas=100, max_iter=10000)
    model.fit(Xz, y)
    return model.coef_, float(model.alpha_)


def _block_bootstrap_indices(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    singers = np.array(sorted(df["singer_id"].unique()))
    sampled = rng.choice(singers, size=len(singers), replace=True)
    indices = []
    for i, singer in enumerate(sampled):
        idx = np.flatnonzero(df["singer_id"].to_numpy() == singer)
        # Keep duplicate sampled singers as duplicate token blocks.
        indices.extend(idx.tolist())
    return np.array(indices, dtype=int)


def lasso_group_stability(df_clean: pd.DataFrame, out_dir: Path, n_bootstrap: int = 200, threshold: float = 0.01):
    rows = []
    fold_rows = []
    for panel, allowed in {"Pressed phonation": ["A", "1"], "Breathy phonation": ["B", "1"]}.items():
        df_sub = df_clean[df_clean["subset_type"].isin(allowed) & df_clean["score_class"].isin(["A", "B", "C"])].dropna(subset=LMM_FEATURES + ["score_label", "singer_id"]).copy()
        X = df_sub[LMM_FEATURES].to_numpy(dtype=float)
        y = df_sub["score_label"].to_numpy(dtype=float)
        full_coef, alpha = _fit_lasso(X, y, seed=42)

        rng = np.random.default_rng(2026)
        boot_records = []
        for b in range(n_bootstrap):
            idx = _block_bootstrap_indices(df_sub, rng)
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                coefs, _ = _fit_lasso(X[idx], y[idx], seed=1000 + b)
            except Exception:
                continue
            for feat, coef in zip(LMM_FEATURES, coefs):
                boot_records.append({"panel": panel, "bootstrap": b + 1, "feature": feat, "coef": float(coef), "selected": abs(coef) >= threshold})

        loso_records = []
        for held_out in sorted(df_sub["singer_id"].unique()):
            train_mask = df_sub["singer_id"].to_numpy() != held_out
            test_mask = ~train_mask
            if train_mask.sum() < 3 or len(np.unique(y[train_mask])) < 2:
                continue
            try:
                coefs, _ = _fit_lasso(X[train_mask], y[train_mask], seed=2000)
                scaler = StandardScaler().fit(X[train_mask])
                train_z = scaler.transform(X[train_mask])
                test_z = scaler.transform(X[test_mask])
                # Refit a LassoCV model for prediction.
                cv = max(2, min(5, train_mask.sum()))
                model = LassoCV(cv=cv, random_state=2000, alphas=100, max_iter=10000)
                model.fit(train_z, y[train_mask])
                pred_cont = model.predict(test_z)
                labels = np.array([1.0, 3.0, 5.0])
                pred_cls = labels[np.argmin(np.abs(pred_cont[:, None] - labels[None, :]), axis=1)]
                acc = accuracy_score(y[test_mask], pred_cls)
            except Exception:
                continue
            for feat, coef in zip(LMM_FEATURES, coefs):
                loso_records.append({"panel": panel, "held_out_singer": held_out, "feature": feat, "coef": float(coef), "selected": abs(coef) >= threshold})
            fold_rows.append(
                {
                    "panel": panel,
                    "held_out_singer": held_out,
                    "n_test": int(test_mask.sum()),
                    "rounded_class_accuracy": float(acc),
                }
            )

        boot_df = pd.DataFrame(boot_records)
        loso_df = pd.DataFrame(loso_records)
        for feat, coef in zip(LMM_FEATURES, full_coef):
            bgrp = boot_df[boot_df["feature"] == feat] if not boot_df.empty else pd.DataFrame()
            lgrp = loso_df[loso_df["feature"] == feat] if not loso_df.empty else pd.DataFrame()
            rows.append(
                {
                    "panel": panel,
                    "feature": feat,
                    "display_feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "full_coef": float(coef),
                    "block_bootstrap_selection_frequency": float(bgrp["selected"].mean()) if not bgrp.empty else np.nan,
                    "block_bootstrap_coef_mean": float(bgrp["coef"].mean()) if not bgrp.empty else np.nan,
                    "block_bootstrap_coef_sd": float(bgrp["coef"].std(ddof=1)) if len(bgrp) > 1 else np.nan,
                    "loso_selection_frequency": float(lgrp["selected"].mean()) if not lgrp.empty else np.nan,
                    "loso_coef_mean": float(lgrp["coef"].mean()) if not lgrp.empty else np.nan,
                    "loso_coef_sd": float(lgrp["coef"].std(ddof=1)) if len(lgrp) > 1 else np.nan,
                    "n_block_bootstrap_success": int(bgrp["bootstrap"].nunique()) if not bgrp.empty else 0,
                    "n_loso_folds": int(lgrp["held_out_singer"].nunique()) if not lgrp.empty else 0,
                    "alpha": alpha,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "lasso_group_stability.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "lasso_loso_prediction.csv", index=False)
    return out


def _permanova(X: np.ndarray, labels: np.ndarray, n_perm: int = 999, seed: int = 42):
    rng = np.random.default_rng(seed)
    d2 = squareform(pdist(X, metric="euclidean")) ** 2
    n = len(labels)
    unique = np.unique(labels)
    ss_total = d2.sum() / n

    def pseudo_f(lab):
        ss_within = 0.0
        for g in np.unique(lab):
            idx = np.flatnonzero(lab == g)
            if len(idx) > 0:
                ss_within += d2[np.ix_(idx, idx)].sum() / len(idx)
        ss_between = ss_total - ss_within
        df_between = len(np.unique(lab)) - 1
        df_within = n - len(np.unique(lab))
        return (ss_between / df_between) / (ss_within / df_within)

    f_obs = pseudo_f(labels)
    perm = np.array([pseudo_f(rng.permutation(labels)) for _ in range(n_perm)])
    p = (np.sum(perm >= f_obs) + 1) / (n_perm + 1)
    return float(f_obs), float(p)


def _mahalanobis_distances(X: np.ndarray, labels: np.ndarray):
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    rows = []
    for i, a in enumerate(["A", "B", "C"]):
        for b in ["A", "B", "C"][i + 1:]:
            xa = X[labels == a]
            xb = X[labels == b]
            if len(xa) == 0 or len(xb) == 0:
                continue
            diff = xa.mean(axis=0) - xb.mean(axis=0)
            dist = math.sqrt(float(diff @ cov_inv @ diff.T))
            rows.append({"group_1": a, "group_2": b, "mahalanobis_distance": dist})
    return pd.DataFrame(rows)


def separation_metrics(df_clean: pd.DataFrame, out_dir: Path):
    rows = []
    mah_rows = []
    for panel, allowed in SUBSET_MAP.items():
        df_sub = df_clean[df_clean["subset_type"].isin(allowed) & df_clean["score_class"].isin(["A", "B", "C"])].dropna(subset=SEPARATION_FEATURES + ["score_class", "singer_id"]).copy()
        if df_sub.empty:
            continue
        X = StandardScaler().fit_transform(df_sub[SEPARATION_FEATURES].to_numpy(dtype=float))
        labels = df_sub["score_class"].to_numpy()
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        f_obs, p_perm = _permanova(X, labels, n_perm=999)

        preds = []
        truths = []
        for held_out in sorted(df_sub["singer_id"].unique()):
            train = df_sub["singer_id"].to_numpy() != held_out
            test = ~train
            if len(np.unique(labels[train])) < 2:
                continue
            scaler = StandardScaler().fit(df_sub.loc[train, SEPARATION_FEATURES])
            x_train = scaler.transform(df_sub.loc[train, SEPARATION_FEATURES])
            x_test = scaler.transform(df_sub.loc[test, SEPARATION_FEATURES])
            clf = LinearDiscriminantAnalysis()
            clf.fit(x_train, labels[train])
            preds.extend(clf.predict(x_test).tolist())
            truths.extend(labels[test].tolist())
        acc = accuracy_score(truths, preds) if truths else np.nan
        bal = balanced_accuracy_score(truths, preds) if truths else np.nan
        macro_f1 = f1_score(truths, preds, average="macro") if truths else np.nan
        rows.append(
            {
                "panel": panel,
                "n": len(df_sub),
                "features": ", ".join(FEATURE_DISPLAY_NAMES[f] for f in SEPARATION_FEATURES),
                "silhouette_score": sil,
                "permanova_pseudo_F": f_obs,
                "permanova_p_value": p_perm,
                "loso_lda_accuracy": acc,
                "loso_lda_balanced_accuracy": bal,
                "loso_lda_macro_f1": macro_f1,
            }
        )
        m = _mahalanobis_distances(X, labels)
        m.insert(0, "panel", panel)
        mah_rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "low_dimensional_separation_metrics.csv", index=False)
    mah = pd.concat(mah_rows, ignore_index=True) if mah_rows else pd.DataFrame()
    mah.to_csv(out_dir / "mahalanobis_distances.csv", index=False)
    return metrics, mah


def lmm_fixed_effects(df_clean: pd.DataFrame, out_dir: Path):
    df = _assignable(df_clean).dropna(subset=["pitch_digit", "score_class", "singer_id"]).copy()
    df["pitch_c"] = df["pitch_digit"].astype(float) - df["pitch_digit"].astype(float).mean()
    rows = []
    sensitivity_rows = []
    for feat in LMM_FEATURES:
        data = df.dropna(subset=[feat]).copy()
        if data.empty:
            continue
        formula = f"Q('{feat}') ~ C(score_class, Treatment(reference='A')) * pitch_c"
        try:
            result = smf.mixedlm(formula, data=data, groups=data["singer_id"]).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            fit_type = "MixedLM"
        except Exception:
            result = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["singer_id"]})
            fit_type = "OLS cluster-robust fallback"
        conf = result.conf_int()
        for term in result.params.index:
            if term == "Intercept" or " Var" in term or term.endswith(" Var"):
                continue
            rows.append(
                {
                    "feature": feat,
                    "display_feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "term": term,
                    "display_term": TERM_DISPLAY_NAMES.get(term, term),
                    "estimate": float(result.params[term]),
                    "std_error": float(result.bse[term]),
                    "statistic": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n": int(result.nobs),
                    "model": fit_type,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = multipletests(out["p_value"].fillna(1.0).to_numpy(), method="fdr_bh")[1]
    out.to_csv(out_dir / "lmm_fixed_effects.csv", index=False)

    for feat, transform_name, transformed in [
        ("Q1", "log1p(Q1)", np.log1p(df["Q1"].astype(float).clip(lower=0))),
    ]:
        data = df.copy()
        data["y_transformed"] = transformed
        formula = "y_transformed ~ C(score_class, Treatment(reference='A')) * pitch_c"
        try:
            result = smf.mixedlm(formula, data=data, groups=data["singer_id"]).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            fit_type = "MixedLM"
        except Exception:
            result = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["singer_id"]})
            fit_type = "OLS cluster-robust fallback"
        conf = result.conf_int()
        for term in result.params.index:
            if term == "Intercept" or " Var" in term or term.endswith(" Var"):
                continue
            sensitivity_rows.append(
                {
                    "feature": feat,
                    "display_feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "transformation": transform_name,
                    "term": term,
                    "display_term": TERM_DISPLAY_NAMES.get(term, term),
                    "estimate": float(result.params[term]),
                    "std_error": float(result.bse[term]),
                    "statistic": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n": int(result.nobs),
                    "model": fit_type,
                }
            )
    sens = pd.DataFrame(sensitivity_rows)
    if not sens.empty:
        sens["p_fdr_bh"] = multipletests(sens["p_value"].fillna(1.0).to_numpy(), method="fdr_bh")[1]
    sens.to_csv(out_dir / "lmm_transform_sensitivity.csv", index=False)
    return out, sens


def ordinal_logit_sensitivity(df_clean: pd.DataFrame, out_dir: Path):
    df = _assignable(df_clean).dropna(subset=SEPARATION_FEATURES + ["score_label", "singer_id"]).copy()
    X = pd.DataFrame(StandardScaler().fit_transform(df[SEPARATION_FEATURES]), columns=SEPARATION_FEATURES, index=df.index)
    singer_dummies = pd.get_dummies(df["singer_id"], prefix="singer", drop_first=True, dtype=float)
    exog = pd.concat([X, singer_dummies], axis=1)
    y = pd.Categorical(df["score_label"].astype(int), categories=[1, 3, 5], ordered=True)
    try:
        model = OrderedModel(y, exog, distr="logit")
        result = model.fit(method="bfgs", maxiter=10000, disp=False)
        conf = result.conf_int()
        rows = []
        for term in exog.columns:
            rows.append(
                {
                    "term": term,
                    "display_term": FEATURE_DISPLAY_NAMES.get(term, term),
                    "coef": float(result.params[term]),
                    "std_error": float(result.bse[term]),
                    "z_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n": len(df),
                }
            )
        out = pd.DataFrame(rows)
        out["p_fdr_bh"] = multipletests(out["p_value"].fillna(1.0).to_numpy(), method="fdr_bh")[1]
        out.to_csv(out_dir / "ordinal_logit_sensitivity.csv", index=False)
        with open(out_dir / "ordinal_logit_summary.txt", "w", encoding="utf-8") as f:
            f.write(result.summary().as_text())
        return out
    except Exception as exc:
        pd.DataFrame([{"error": str(exc)}]).to_csv(out_dir / "ordinal_logit_sensitivity.csv", index=False)
        return pd.DataFrame()


def write_markdown_summary(out_dir: Path):
    def load(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    flow = load("data_flow_summary.csv")
    sep = load("low_dimensional_separation_metrics.csv")
    lasso = load("lasso_group_stability.csv")
    with open(out_dir / "revision_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("Revision analysis summary\n")
        f.write("=========================\n\n")
        if not flow.empty:
            f.write("Data flow:\n")
            for _, r in flow.iterrows():
                f.write(f"- {r['stage']}: n={r['n']} ({r['note']})\n")
            f.write("\n")
        if not sep.empty:
            f.write("Low-dimensional separation metrics:\n")
            for _, r in sep.iterrows():
                f.write(
                    f"- {r['panel']}: n={int(r['n'])}, silhouette={r['silhouette_score']:.3f}, "
                    f"PERMANOVA F={r['permanova_pseudo_F']:.2f}, p={_format_p(r['permanova_p_value'])}, "
                    f"LOSO LDA accuracy={r['loso_lda_accuracy']:.3f}, macro-F1={r['loso_lda_macro_f1']:.3f}\n"
                )
            f.write("\n")
        if not lasso.empty:
            f.write("Selected LASSO stability (block bootstrap / LOSO):\n")
            key = lasso[(lasso["feature"].isin(["HNR", "CPP", "Q1", "H1H2_output"]))]
            for _, r in key.iterrows():
                f.write(
                    f"- {r['panel']} {r['display_feature']}: coef={r['full_coef']:.3f}, "
                    f"block={r['block_bootstrap_selection_frequency']:.3f}, "
                    f"LOSO={r['loso_selection_frequency']:.3f}\n"
                )
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="all-CUN")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    repo_root = Path.cwd()
    outputs_root = repo_root / args.outputs_root / args.dataset_name
    data_dir = repo_root / args.data_root / args.dataset_name
    out_dir = outputs_root / "revision_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = _load_feature_data(outputs_root)
    df_clean, outlier_name, outlier_jitter = _remove_max_jitter(df_raw)
    df_clean.to_csv(out_dir / "analysis_tokens_after_outlier.csv")

    print("[*] Writing data-flow tables")
    make_data_flow_tables(df_raw, df_clean, data_dir, out_dir, outlier_name, outlier_jitter)

    print("[*] Running singer-level LASSO stability")
    lasso_group_stability(df_clean, out_dir)

    print("[*] Running low-dimensional separation metrics")
    separation_metrics(df_clean, out_dir)

    print("[*] Running LMM fixed-effect tables")
    lmm_fixed_effects(df_clean, out_dir)

    print("[*] Running ordinal-logit sensitivity analysis")
    ordinal_logit_sensitivity(df_clean, out_dir)

    write_markdown_summary(out_dir)
    print(f"[+] Revision analysis complete: {out_dir}")


if __name__ == "__main__":
    main()
