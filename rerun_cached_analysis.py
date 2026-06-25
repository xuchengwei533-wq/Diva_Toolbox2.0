import argparse
from pathlib import Path

import pandas as pd

from src.combined_data import CombinedData
from src.data_parser import parse_label
from src.models import run_lasso_analysis
from src.plot_scatter import (
    generate_paper_2d_joint_horizontal,
    generate_paper_3d_ellipsoid_horizontal,
    plot_scatter_ndim,
)
from src.utils.config_loader import load_config


def _build_filename_score_matrix(audio_filenames, tech_name="chest"):
    rows = []
    for audio_filename in audio_filenames:
        label = parse_label(audio_filename)
        if label is not None:
            rows.append({"audio_filename": audio_filename, tech_name: int(label)})
    if not rows:
        raise ValueError("No valid score labels could be parsed from audio filenames.")
    df_score = pd.DataFrame(rows).drop_duplicates(subset=["audio_filename"]).set_index("audio_filename")
    df_score.index.name = "audio_filename"
    return df_score


def _load_cached_feature_stats(dataset_output_root: Path):
    cached_summary = dataset_output_root / "feats_data.csv"
    if not cached_summary.exists():
        raise FileNotFoundError(f"Cached feature summary not found: {cached_summary}")
    df_stats = pd.read_csv(cached_summary, index_col=0)
    df_stats.index.name = "audio_filename"
    return df_stats


def _subset_groups_from_config(cfg):
    if "dataset" not in cfg or "subset_groups" not in cfg.dataset or not cfg.dataset.subset_groups:
        return [["A", "1"], ["B", "1"], ["A", "B", "1"]]
    return [[str(x) for x in group] for group in cfg.dataset.subset_groups]


def rerun_cached_analysis(project_root: Path, dataset_name: str, config_path: Path):
    cfg = load_config(str(config_path))
    dataset_output_root = project_root / "outputs" / dataset_name
    analysis_dir = dataset_output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Dataset: {dataset_name}")
    print(f"[*] Reusing cached feature summary under: {dataset_output_root}")
    df_stats_all = _load_cached_feature_stats(dataset_output_root)
    if df_stats_all.empty:
        raise ValueError("Cached feature stats are empty.")

    feat_names = list(cfg.acoustic_feats) if "acoustic_feats" in cfg else list(df_stats_all.columns)
    feat_names = [feat for feat in feat_names if feat in df_stats_all.columns]
    missing_feats = [feat for feat in list(cfg.acoustic_feats) if feat not in df_stats_all.columns] if "acoustic_feats" in cfg else []
    if missing_feats:
        print(f"[!] Configured features missing from cached table and skipped: {missing_feats}")
    df_stats = df_stats_all[feat_names].copy()

    df_score = _build_filename_score_matrix(list(df_stats_all.index), tech_name="chest")
    combined = CombinedData(df_score, df_stats)
    combined.save_to_csv(str(dataset_output_root), filename=f"score_feats_data_{dataset_name}.csv")

    print("[*] Rebuilding scatter plots")
    for ndim in [1, 2, 3]:
        plot_scatter_ndim(df_stats, str(dataset_output_root), ndim, feat_names=feat_names)

    print("[*] Rebuilding paper-style plots")
    for x_col, y_col in [
        ("H1H2_output", "CPP"),
        ("H1H2_output", "Q1"),
        ("H1H2_output", "HNR"),
    ]:
        if x_col in df_stats.columns and y_col in df_stats.columns:
            generate_paper_2d_joint_horizontal(
                df_stats,
                str(dataset_output_root),
                x_col=x_col,
                y_col=y_col,
                pitch_name="chest",
            )

    for x_col, y_col, z_col in [
        ("H1H2_output", "Q1", "CPP"),
        ("H1H2_output", "HNR", "CPP"),
    ]:
        if all(col in df_stats.columns for col in [x_col, y_col, z_col]):
            generate_paper_3d_ellipsoid_horizontal(
                df_stats,
                str(dataset_output_root),
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                pitch_name="chest",
            )

    print("[*] Rebuilding LASSO outputs")
    for subsets in _subset_groups_from_config(cfg):
        run_lasso_analysis(combined, "chest", str(analysis_dir), subsets)

    print("[+] Cached analysis rebuild complete.")
    print(f"    plots: {dataset_output_root}")
    print(f"    lasso: {analysis_dir / 'chest'}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild plots and LASSO outputs from cached feats_data.csv.")
    parser.add_argument("--dataset-name", required=True, help="Dataset folder under outputs/.")
    parser.add_argument("--config", default="configs/basic_cfg.yaml", help="YAML config path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    rerun_cached_analysis(project_root, args.dataset_name, config_path)


if __name__ == "__main__":
    main()
