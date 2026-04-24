import argparse
import os
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, List, Optional

# 强制使用无界面后端，避免 Tk 主线程与后台绘图线程在退出时冲突。
os.environ.setdefault("MPLBACKEND", "Agg")

from src.combined_data import CombinedData
from src.data_loader import load_score_matrix
from src.feat_extract.feat_extractor import extract_feats_from_wav_dir, extract_feats_stats_from_csv
from src.models import (
    run_correlation_matrix,
    run_lasso_analysis,
    run_lasso_correlation_matrix,
    run_ordinal_correlation_matrix,
)
from src.plot_scatter import plot_scatter_ndim
from src.regression import run_ordinal_regression_with_stratified_kfold_cv
from src.utils.config_loader import load_config


def _resolve_score_file(dataset_dir: Path, score_file_arg: Optional[str], score_file_cfg: Optional[str]) -> Path:
    if score_file_arg:
        candidate = dataset_dir / score_file_arg
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Score file not found: {candidate}")

    if score_file_cfg:
        candidate = dataset_dir / str(score_file_cfg)
        if candidate.exists():
            return candidate

    excel_candidates = sorted(list(dataset_dir.glob("*scores*.xlsx")) + list(dataset_dir.glob("*.xlsx")))
    if len(excel_candidates) == 1:
        return excel_candidates[0]
    if len(excel_candidates) == 0:
        raise FileNotFoundError(f"No .xlsx score file found under {dataset_dir}")
    raise FileNotFoundError(
        f"Multiple .xlsx score files found under {dataset_dir}. "
        f"Please provide --score-file. Candidates: {[p.name for p in excel_candidates]}"
    )


def _normalize_subset_groups(raw_subset_groups) -> List[List[str]]:
    if not raw_subset_groups:
        return [["A", "1"], ["B", "1"], ["A", "B", "1"]]
    groups = []
    for group in raw_subset_groups:
        groups.append([str(x) for x in group])
    return groups


def _matrix_dir_name(subsets: List[str]) -> str:
    key = "".join(subsets)
    if key == "A1":
        return "matrix_A1"
    if key == "B1":
        return "matrix_B1"
    if key == "AB1":
        return "matrix"
    return f"matrix_{key}"


def _list_dataset_names(project_root: Path) -> List[str]:
    data_dir = project_root / "data"
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def _list_score_files(project_root: Path, dataset_name: str) -> List[str]:
    dataset_dir = project_root / "data" / dataset_name
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.glob("*.xlsx")])


def run_pipeline(
        project_root: Path,
        cfg,
        dataset_name: str,
        score_file_name: Optional[str],
        n_splits: int,
        overwrite: bool,
        visualize: bool,
        log: Callable[[str], None] = print,
):
    os.chdir(project_root)
    log(f"[*] Project root: {project_root}")

    dataset_dir = project_root / "data" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    cfg_score_file = None
    if "dataset" in cfg and "score_file" in cfg.dataset:
        cfg_score_file = cfg.dataset.score_file
    score_path = None
    try:
        score_path = _resolve_score_file(dataset_dir, score_file_name, cfg_score_file)
    except FileNotFoundError:
        log("[!] 未找到评分文件：将进入无监督模式（仅提特征与绘图，跳过回归/矩阵分析）。")

    log(f"[*] Dataset: {dataset_name}")
    if score_path is not None:
        log(f"[*] Score file: {score_path.name}")
    else:
        log("[*] Score file: None")

    outputs_root = project_root / "outputs"
    dataset_output_root = outputs_root / dataset_name

    raw_feats_dir = dataset_output_root / "raw_feats"
    analysis_dir = dataset_output_root / "analysis"
    regression_dir = dataset_output_root / "regression_kfold"

    raw_feats_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)

    log("[*] Step 1/6: feature extraction")
    extract_feats_from_wav_dir(
        wav_dir=str(dataset_dir),
        output_dir=str(raw_feats_dir),
        visualize=visualize,
        overwrite=overwrite,
    )

    log("[*] Step 2/6: build merged table")
    df_stats = extract_feats_stats_from_csv(str(raw_feats_dir), str(dataset_output_root))
    if df_stats.empty:
        raise ValueError("Extracted feature stats are empty.")

    combined = None
    if score_path is not None:
        df_score = load_score_matrix(str(score_path))
        if df_score.empty:
            raise ValueError("Loaded score matrix is empty.")
        combined = CombinedData(df_score, df_stats)
        combined.save_to_csv(str(dataset_output_root), filename=f"score_feats_data_{dataset_name}.csv")
    else:
        log("[*] 跳过评分表合并（无评分文件）。")

    log("[*] Step 3/6: scatter plots (1D, 2D, 3D)")
    feat_names = list(cfg.acoustic_feats) if "acoustic_feats" in cfg else None
    for ndim in [1, 2, 3]:
        plot_scatter_ndim(df_stats, str(dataset_output_root), ndim, feat_names=feat_names)

    if combined is not None:
        log("[*] Step 4/6: LASSO analysis")
        subset_groups = _normalize_subset_groups(cfg.dataset.subset_groups if "dataset" in cfg else None)
        target_techs = list(cfg.vocal_techs) if "vocal_techs" in cfg else list(combined.tech_cols)
        for tech in target_techs:
            for subsets in subset_groups:
                run_lasso_analysis(combined, tech, str(analysis_dir), subsets)

        log("[*] Step 5/6: stratified K-fold ordinal regression")
        for tech in target_techs:
            for subsets in subset_groups:
                run_ordinal_regression_with_stratified_kfold_cv(
                    combined,
                    tech,
                    str(regression_dir),
                    subsets=subsets,
                    n_splits=n_splits,
                )

        log("[*] Step 6/6: matrix outputs (matrix, matrix_A1, matrix_B1)")
        for subsets in subset_groups:
            matrix_dir = dataset_output_root / _matrix_dir_name(subsets)
            matrix_dir.mkdir(parents=True, exist_ok=True)
            run_correlation_matrix(combined, str(matrix_dir), subsets)
            run_lasso_correlation_matrix(combined, str(matrix_dir), subsets)
            run_ordinal_correlation_matrix(combined, str(matrix_dir), subsets, metric="coef")
            run_ordinal_correlation_matrix(combined, str(matrix_dir), subsets, metric="or")
    else:
        log("[*] Step 4/6: 跳过（无评分文件）")
        log("[*] Step 5/6: 跳过（无评分文件）")
        log("[*] Step 6/6: 跳过（无评分文件）")

    log("[+] Full pipeline complete.")
    log(f"    dataset outputs root: {dataset_output_root}")
    log(f"    scatter: {dataset_output_root / 'plot_1d'}")
    log(f"    scatter: {dataset_output_root / 'plot_2d'}")
    log(f"    scatter: {dataset_output_root / 'plot_3d'}")
    if combined is not None:
        log(f"    matrix: {dataset_output_root / 'matrix'}")
        log(f"    matrix: {dataset_output_root / 'matrix_A1'}")
        log(f"    matrix: {dataset_output_root / 'matrix_B1'}")


class PipelineGUI:
    def __init__(self, root: Tk, project_root: Path, cfg):
        self.root = root
        self.project_root = project_root
        self.cfg = cfg
        self.root.title("Acoustics Score Analysis - Full Pipeline")
        self.root.geometry("860x620")

        self.dataset_var = StringVar()
        self.score_file_var = StringVar()
        self.n_splits_var = IntVar(value=10)
        self.overwrite_var = BooleanVar(value=False)
        self.visualize_var = BooleanVar(value=False)

        self._build_ui()
        self._refresh_datasets()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="数据集（data 下文件夹）:").grid(row=0, column=0, sticky="w")
        self.dataset_combo = ttk.Combobox(frame, textvariable=self.dataset_var, state="readonly", width=48)
        self.dataset_combo.grid(row=0, column=1, sticky="we", padx=8)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_score_files())

        ttk.Button(frame, text="刷新数据集", command=self._refresh_datasets).grid(row=0, column=2, padx=6)

        ttk.Label(frame, text="评分表 (.xlsx):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.score_combo = ttk.Combobox(frame, textvariable=self.score_file_var, state="readonly", width=48)
        self.score_combo.grid(row=1, column=1, sticky="we", padx=8, pady=(8, 0))
        ttk.Button(frame, text="刷新评分表", command=self._refresh_score_files).grid(row=1, column=2, padx=6, pady=(8, 0))

        ttk.Label(frame, text="分层交叉验证折数:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(frame, from_=2, to=20, textvariable=self.n_splits_var, width=10).grid(
            row=2, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        ttk.Checkbutton(frame, text="覆盖已有特征文件（overwrite）", variable=self.overwrite_var).grid(
            row=3, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Checkbutton(frame, text="保存特征序列可视化（visualize）", variable=self.visualize_var).grid(
            row=4, column=1, sticky="w", padx=8
        )

        self.run_btn = ttk.Button(frame, text="一键运行全流程", command=self._on_run_clicked)
        self.run_btn.grid(row=5, column=1, sticky="w", padx=8, pady=(10, 6))

        ttk.Label(frame, text="运行日志:").grid(row=6, column=0, sticky="nw")
        self.log_text = ScrolledText(frame, wrap="word", height=24)
        self.log_text.grid(row=6, column=1, columnspan=2, sticky="nsew", padx=8)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)

    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def _refresh_datasets(self):
        datasets = _list_dataset_names(self.project_root)
        self.dataset_combo["values"] = datasets
        if datasets:
            if self.dataset_var.get() not in datasets:
                default_name = self.cfg.dataset.name if "dataset" in self.cfg and "name" in self.cfg.dataset else datasets[0]
                self.dataset_var.set(default_name if default_name in datasets else datasets[0])
            self._refresh_score_files()
        else:
            self.dataset_var.set("")
            self.score_combo["values"] = []
            self.score_file_var.set("")

    def _refresh_score_files(self):
        dataset_name = self.dataset_var.get().strip()
        score_files = _list_score_files(self.project_root, dataset_name) if dataset_name else []
        self.score_combo["values"] = score_files
        if score_files:
            if self.score_file_var.get() not in score_files:
                cfg_score = self.cfg.dataset.score_file if "dataset" in self.cfg and "score_file" in self.cfg.dataset else score_files[0]
                self.score_file_var.set(cfg_score if cfg_score in score_files else score_files[0])
        else:
            self.score_file_var.set("")

    def _set_running(self, running: bool):
        self.run_btn.config(state="disabled" if running else "normal")
        self.dataset_combo.config(state="disabled" if running else "readonly")
        self.score_combo.config(state="disabled" if running else "readonly")

    def _on_run_clicked(self):
        dataset_name = self.dataset_var.get().strip()
        score_file_name = self.score_file_var.get().strip() or None
        if not dataset_name:
            messagebox.showerror("错误", "请先选择数据集。")
            return

        self._set_running(True)
        self._append_log("=" * 72)
        self._append_log("[*] 开始运行全流程...")

        def worker():
            try:
                run_pipeline(
                    project_root=self.project_root,
                    cfg=self.cfg,
                    dataset_name=dataset_name,
                    score_file_name=score_file_name,
                    n_splits=int(self.n_splits_var.get()),
                    overwrite=bool(self.overwrite_var.get()),
                    visualize=bool(self.visualize_var.get()),
                    log=lambda s: self.root.after(0, self._append_log, s),
                )
                self.root.after(0, lambda: messagebox.showinfo("完成", "全流程运行完成。"))
            except Exception as e:
                err_msg = str(e)
                err_trace = traceback.format_exc()
                self.root.after(0, self._append_log, f"[!] 运行失败: {err_msg}")
                self.root.after(0, self._append_log, err_trace)
                self.root.after(0, lambda msg=err_msg: messagebox.showerror("运行失败", msg))
            finally:
                self.root.after(0, self._set_running, False)

        threading.Thread(target=worker, daemon=True).start()


def _run_cli(args):
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = load_config(str(config_path))
    dataset_name = args.dataset_name or input("Enter dataset folder under data/: ").strip()
    if not dataset_name:
        raise ValueError("dataset name cannot be empty.")

    run_pipeline(
        project_root=project_root,
        cfg=cfg,
        dataset_name=dataset_name,
        score_file_name=args.score_file,
        n_splits=args.n_splits,
        overwrite=args.overwrite,
        visualize=args.visualize,
    )


def _run_gui(config_path: str):
    project_root = Path(__file__).resolve().parent
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = project_root / resolved_config
    cfg = load_config(str(resolved_config))
    root = Tk()
    app = PipelineGUI(root=root, project_root=project_root, cfg=cfg)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Acoustics full pipeline (GUI by default).")
    parser.add_argument("--config", default="configs/basic_cfg.yaml", help="YAML config path.")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI.")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder under data/.")
    parser.add_argument("--score-file", default=None, help="Score matrix filename under dataset folder.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for stratified CV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw feature files.")
    parser.add_argument("--visualize", action="store_true", help="Save feature-sequence plots during extraction.")
    args = parser.parse_args()

    if args.cli:
        _run_cli(args)
    else:
        _run_gui(args.config)


if __name__ == "__main__":
    main()
