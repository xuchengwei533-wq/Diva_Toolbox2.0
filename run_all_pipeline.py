import argparse
import os
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, END, EXTENDED, IntVar, StringVar, Tk, Listbox, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, List, Optional

import pandas as pd

# 强制使用无界面后端，避免 Tk 主线程与后台绘图线程在退出时冲突。
os.environ.setdefault("MPLBACKEND", "Agg")

from src.combined_data import CombinedData
from src.data_loader import load_score_matrix
from src.data_parser import parse_label
from src.feat_extract.feat_extractor import extract_feats_from_wav_dir, extract_feats_stats_from_csv
from src.models import (
    run_correlation_matrix,
    run_lasso_analysis,
    run_lasso_correlation_matrix,
    run_ordinal_correlation_matrix,
)
from src.plot_scatter import (
    generate_paper_2d_joint_horizontal,
    generate_paper_3d_ellipsoid_horizontal,
    plot_scatter_ndim,
)
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
    if subsets == ["A", "1"]:
        return "matrix_pressed_phonation"
    if subsets == ["B", "1"]:
        return "matrix_breathy_phonation"
    if subsets == ["A", "B", "1"]:
        return "matrix"
    key = "".join(subsets)
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


def _has_existing_feature_cache(raw_feats_dir: Path) -> bool:
    if not raw_feats_dir.exists():
        return False
    return any(raw_feats_dir.rglob("*.csv"))


def _load_existing_feature_stats(dataset_output_root: Path, raw_feats_dir: Path, log: Callable[[str], None]) -> pd.DataFrame:
    cached_summary = dataset_output_root / "feats_data.csv"
    if cached_summary.exists():
        log(f"[*] Reusing cached feature summary: {cached_summary}")
        df_stats = pd.read_csv(cached_summary, index_col=0)
        df_stats.index.name = "audio_filename"
        return df_stats

    log("[*] Cached summary not found, rebuilding stats from existing raw feature CSV files.")
    return extract_feats_stats_from_csv(str(raw_feats_dir), str(dataset_output_root))


def _build_filename_score_matrix(audio_filenames: List[str], tech_name: str = "chest"):
    rows = []
    for audio_filename in audio_filenames:
        label = parse_label(audio_filename)
        if label is None:
            continue
        rows.append({"audio_filename": audio_filename, tech_name: int(label)})

    if not rows:
        raise ValueError("No valid score labels could be parsed from audio filenames.")

    df_score = pd.DataFrame(rows).drop_duplicates(subset=["audio_filename"]).set_index("audio_filename")
    df_score.index.name = "audio_filename"
    return df_score


def run_pipeline(
        project_root: Path,
        cfg,
        dataset_name: str,
        score_file_name: Optional[str],
        n_splits: int,
        overwrite: bool,
        visualize: bool,
        reuse_existing: bool,
        log: Callable[[str], None] = print,
        progress_callback: Optional[Callable[[float, str], None]] = None,
):
    os.chdir(project_root)
    log(f"[*] Project root: {project_root}")

    dataset_dir = project_root / "data" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    log(f"[*] Dataset: {dataset_name}")
    log("[*] Score source: filename labels")
    if score_file_name:
        log("[*] Note: 当前流程已不使用评分表，--score-file 将被忽略。")

    outputs_root = project_root / "outputs"
    dataset_output_root = outputs_root / dataset_name

    raw_feats_dir = dataset_output_root / "raw_feats"
    analysis_dir = dataset_output_root / "analysis"
    regression_dir = dataset_output_root / "regression_kfold"

    raw_feats_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(0.0, f"{dataset_name}: 准备开始")

    reuse_hit = reuse_existing and _has_existing_feature_cache(raw_feats_dir)
    if reuse_hit:
        log("[*] Step 1/6: reuse existing extracted features")
        if overwrite:
            log("[*] Note: 已启用复用模式，本次不会重新提取特征，overwrite 将被忽略。")
        if progress_callback is not None:
            progress_callback(70.0, f"{dataset_name}: 已复用现有特征文件")
    else:
        if reuse_existing:
            log("[*] Step 1/6: feature extraction")
            log("[*] 未找到可复用的已提取特征，将执行新的特征提取。")
        else:
            log("[*] Step 1/6: feature extraction")
        extract_feats_from_wav_dir(
            wav_dir=str(dataset_dir),
            output_dir=str(raw_feats_dir),
            visualize=visualize,
            overwrite=overwrite,
            log_callback=log,
            progress_callback=(
                lambda current, total, rel_path, status: progress_callback(
                    (current / max(total, 1)) * 70.0,
                    f"{dataset_name}: 特征提取 {current}/{total} | {rel_path or '准备中'} | {status}",
                )
                if progress_callback is not None
                else None
            ),
        )

    log("[*] Step 2/6: build filename-derived chest scores + merged table")
    if reuse_hit:
        df_stats = _load_existing_feature_stats(dataset_output_root, raw_feats_dir, log)
    else:
        df_stats = extract_feats_stats_from_csv(str(raw_feats_dir), str(dataset_output_root))
    if df_stats.empty:
        raise ValueError("Extracted feature stats are empty.")
    if progress_callback is not None:
        progress_callback(80.0, f"{dataset_name}: 已完成特征汇总")

    df_score = _build_filename_score_matrix(list(df_stats.index), tech_name="chest")
    combined = CombinedData(df_score, df_stats)
    combined.save_to_csv(str(dataset_output_root), filename=f"score_feats_data_{dataset_name}.csv")

    log("[*] Step 3/6: scatter plots (1D, 2D, 3D)")
    feat_names = list(cfg.acoustic_feats) if "acoustic_feats" in cfg else None
    for ndim in [1, 2, 3]:
        plot_scatter_ndim(df_stats, str(dataset_output_root), ndim, feat_names=feat_names)
    log("[*] Step 3b/6: paper-style joint and ellipsoid plots")
    paper_2d_pairs = [
        ("H1H2_output", "CPP"),
        ("H1H2_output", "Q1"),
        ("H1H2_output", "HNR"),
    ]
    paper_3d_triplets = [
        ("H1H2_output", "Q1", "CPP"),
        ("H1H2_output", "HNR", "CPP"),
    ]
    for x_col, y_col in paper_2d_pairs:
        if x_col in df_stats.columns and y_col in df_stats.columns:
            generate_paper_2d_joint_horizontal(
                df_stats,
                str(dataset_output_root),
                x_col=x_col,
                y_col=y_col,
                pitch_name="chest",
            )
    for x_col, y_col, z_col in paper_3d_triplets:
        if all(col in df_stats.columns for col in [x_col, y_col, z_col]):
            generate_paper_3d_ellipsoid_horizontal(
                df_stats,
                str(dataset_output_root),
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                pitch_name="chest",
            )
    if progress_callback is not None:
        progress_callback(90.0, f"{dataset_name}: 已完成散点图")

    log("[*] Step 4/6: LASSO analysis (chest only)")
    subset_groups = _normalize_subset_groups(cfg.dataset.subset_groups if "dataset" in cfg else None)
    for subsets in subset_groups:
        run_lasso_analysis(combined, "chest", str(analysis_dir), subsets)
    if progress_callback is not None:
        progress_callback(100.0, f"{dataset_name}: 已完成 chest LASSO")

    log("[*] Step 5/6: 跳过（当前仅保留 chest 的 LASSO）")
    log("[*] Step 6/6: 跳过（当前仅保留 chest 的 LASSO）")

    log("[+] Full pipeline complete.")
    log(f"    dataset outputs root: {dataset_output_root}")
    log(f"    scatter: {dataset_output_root / 'plot_1d'}")
    log(f"    scatter: {dataset_output_root / 'plot_2d'}")
    log(f"    scatter: {dataset_output_root / 'plot_3d'}")
    log(f"    lasso: {analysis_dir / 'chest'}")


def run_pipelines(
        project_root: Path,
        cfg,
        dataset_names: List[str],
        score_file_name: Optional[str],
        n_splits: int,
        overwrite: bool,
        visualize: bool,
        reuse_existing: bool,
        log: Callable[[str], None] = print,
        progress_callback: Optional[Callable[[float, str], None]] = None,
):
    if not dataset_names:
        raise ValueError("dataset names cannot be empty.")
    if score_file_name and len(dataset_names) != 1:
        raise ValueError("score_file_name only supports a single dataset run.")

    total = len(dataset_names)
    for idx, dataset_name in enumerate(dataset_names, start=1):
        log("=" * 72)
        log(f"[*] 开始处理数据集 {idx}/{total}: {dataset_name}")
        dataset_base = ((idx - 1) / total) * 100.0
        dataset_span = 100.0 / total
        run_pipeline(
            project_root=project_root,
            cfg=cfg,
            dataset_name=dataset_name,
            score_file_name=score_file_name if total == 1 else None,
            n_splits=n_splits,
            overwrite=overwrite,
            visualize=visualize,
            reuse_existing=reuse_existing,
            log=log,
            progress_callback=(
                lambda pct, message, base=dataset_base, span=dataset_span, order=idx, total_count=total:
                progress_callback(
                    base + (pct / 100.0) * span,
                    f"[{order}/{total_count}] {message}",
                )
                if progress_callback is not None
                else None
            ),
        )

    log("=" * 72)
    log(f"[+] 批量流程完成，共处理 {total} 个数据集。")


class PipelineGUI:
    def __init__(self, root: Tk, project_root: Path, cfg):
        self.root = root
        self.project_root = project_root
        self.cfg = cfg
        self.root.title("Acoustics Score Analysis - Full Pipeline")
        self.root.geometry("980x700")

        self.dataset_hint_var = StringVar(value="请选择 1 个或多个数据集。")
        self.progress_status_var = StringVar(value="就绪")
        self.score_file_var = StringVar()
        self.progress_var = DoubleVar(value=0.0)
        self.n_splits_var = IntVar(value=10)
        self.overwrite_var = BooleanVar(value=False)
        self.visualize_var = BooleanVar(value=False)
        self.reuse_existing_var = BooleanVar(value=True)
        self.dataset_names: List[str] = []

        self._build_ui()
        self._refresh_datasets()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="数据集（data 下文件夹，可多选）:").grid(row=0, column=0, sticky="nw")
        dataset_frame = ttk.Frame(frame)
        dataset_frame.grid(row=0, column=1, sticky="nsew", padx=8)
        self.dataset_listbox = Listbox(
            dataset_frame,
            selectmode=EXTENDED,
            exportselection=False,
            height=8,
        )
        self.dataset_listbox.grid(row=0, column=0, sticky="nsew")
        dataset_scrollbar = ttk.Scrollbar(dataset_frame, orient="vertical", command=self.dataset_listbox.yview)
        dataset_scrollbar.grid(row=0, column=1, sticky="ns")
        self.dataset_listbox.config(yscrollcommand=dataset_scrollbar.set)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selection_changed)
        ttk.Label(dataset_frame, textvariable=self.dataset_hint_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        dataset_frame.columnconfigure(0, weight=1)
        dataset_frame.rowconfigure(0, weight=1)

        dataset_btns = ttk.Frame(frame)
        dataset_btns.grid(row=0, column=2, sticky="n", padx=6)
        self.select_all_btn = ttk.Button(dataset_btns, text="全选", command=self._select_all_datasets)
        self.select_all_btn.grid(row=0, column=0, sticky="we")
        self.clear_selection_btn = ttk.Button(dataset_btns, text="清空", command=self._clear_dataset_selection)
        self.clear_selection_btn.grid(row=1, column=0, sticky="we", pady=6)
        self.refresh_btn = ttk.Button(dataset_btns, text="刷新数据集", command=self._refresh_datasets)
        self.refresh_btn.grid(row=2, column=0, sticky="we")

        ttk.Label(frame, text="评分表（单数据集可手动指定）:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.score_combo = ttk.Combobox(frame, textvariable=self.score_file_var, state="disabled", width=48)
        self.score_combo.grid(row=1, column=1, sticky="we", padx=8, pady=(8, 0))
        self.refresh_scores_btn = ttk.Button(frame, text="刷新评分表", command=self._refresh_score_files)
        self.refresh_scores_btn.grid(row=1, column=2, padx=6, pady=(8, 0))

        ttk.Label(frame, text="评分表预览:").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        self.score_preview_text = ScrolledText(frame, wrap="word", height=6)
        self.score_preview_text.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=8, pady=(8, 0))
        self.score_preview_text.config(state="disabled")

        ttk.Label(frame, text="分层交叉验证折数:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(frame, from_=2, to=20, textvariable=self.n_splits_var, width=10).grid(
            row=3, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        ttk.Checkbutton(frame, text="覆盖已有特征文件（overwrite）", variable=self.overwrite_var).grid(
            row=4, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Checkbutton(frame, text="保存特征序列可视化（visualize）", variable=self.visualize_var).grid(
            row=5, column=1, sticky="w", padx=8
        )
        ttk.Checkbutton(frame, text="优先复用已提取特征（跳过重复提取）", variable=self.reuse_existing_var).grid(
            row=6, column=1, sticky="w", padx=8
        )

        self.run_btn = ttk.Button(frame, text="运行所选数据集", command=self._on_run_clicked)
        self.run_btn.grid(row=7, column=1, sticky="w", padx=8, pady=(10, 6))

        ttk.Label(frame, text="运行进度:").grid(row=8, column=0, sticky="w")
        progress_frame = ttk.Frame(frame)
        progress_frame.grid(row=8, column=1, columnspan=2, sticky="we", padx=8)
        ttk.Label(progress_frame, textvariable=self.progress_status_var).grid(row=0, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=1, column=0, sticky="we", pady=(4, 0))
        progress_frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="运行日志:").grid(row=9, column=0, sticky="nw")
        self.log_text = ScrolledText(frame, wrap="word", height=24)
        self.log_text.grid(row=9, column=1, columnspan=2, sticky="nsew", padx=8)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(2, weight=0)
        frame.rowconfigure(9, weight=2)

    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def _update_progress_ui(self, value: float, status: str):
        self.progress_var.set(max(0.0, min(100.0, value)))
        self.progress_status_var.set(status)
        self.root.update_idletasks()

    def _get_selected_dataset_names(self) -> List[str]:
        return [self.dataset_listbox.get(i) for i in self.dataset_listbox.curselection()]

    def _set_selected_datasets(self, selected_names: List[str]):
        self.dataset_listbox.selection_clear(0, END)
        wanted = set(selected_names)
        for idx, name in enumerate(self.dataset_names):
            if name in wanted:
                self.dataset_listbox.selection_set(idx)
        self._refresh_score_files()

    def _set_score_preview(self, lines: List[str]):
        self.score_preview_text.config(state="normal")
        self.score_preview_text.delete("1.0", "end")
        self.score_preview_text.insert("1.0", "\n".join(lines))
        self.score_preview_text.config(state="disabled")

    def _on_dataset_selection_changed(self, _event=None):
        self._refresh_score_files()

    def _select_all_datasets(self):
        self.dataset_listbox.selection_set(0, END)
        self._refresh_score_files()

    def _clear_dataset_selection(self):
        self.dataset_listbox.selection_clear(0, END)
        self._refresh_score_files()

    def _refresh_datasets(self):
        previous_selection = set(self._get_selected_dataset_names())
        self.dataset_names = _list_dataset_names(self.project_root)
        self.dataset_listbox.delete(0, END)
        for name in self.dataset_names:
            self.dataset_listbox.insert(END, name)

        if self.dataset_names:
            default_name = (
                self.cfg.dataset.name
                if "dataset" in self.cfg and "name" in self.cfg.dataset
                else self.dataset_names[0]
            )
            selection = list(previous_selection.intersection(self.dataset_names))
            if not selection:
                selection = [default_name if default_name in self.dataset_names else self.dataset_names[0]]
            self._set_selected_datasets(selection)
        else:
            self.score_combo["values"] = []
            self.score_file_var.set("")
            self.score_combo.config(state="disabled")
            self._set_score_preview(["未找到任何数据集。"])
            self.dataset_hint_var.set("未找到 data 下的数据集文件夹。")

    def _refresh_score_files(self):
        dataset_names = self._get_selected_dataset_names()
        if not dataset_names:
            self.score_combo["values"] = []
            self.score_file_var.set("")
            self.score_combo.config(state="disabled")
            self.dataset_hint_var.set("请选择至少 1 个数据集。")
            self._set_score_preview(["未选择数据集。"])
            return

        self.dataset_hint_var.set(
            f"已选择 {len(dataset_names)} 个数据集：{', '.join(dataset_names)}"
        )

        preview_lines = []
        if len(dataset_names) == 1:
            dataset_name = dataset_names[0]
            score_files = _list_score_files(self.project_root, dataset_name)
            self.score_combo["values"] = score_files
            if score_files:
                self.score_combo.config(state="readonly")
                if self.score_file_var.get() not in score_files:
                    cfg_score = self.cfg.dataset.score_file if "dataset" in self.cfg and "score_file" in self.cfg.dataset else score_files[0]
                    self.score_file_var.set(cfg_score if cfg_score in score_files else score_files[0])
                preview_lines.append(f"{dataset_name}: 可选评分表 {score_files}")
                preview_lines.append("运行时将优先使用上方手动指定的评分表。")
            else:
                self.score_file_var.set("")
                self.score_combo.config(state="disabled")
                preview_lines.append(f"{dataset_name}: 未发现 .xlsx，将自动进入无监督模式。")
        else:
            self.score_combo["values"] = []
            self.score_file_var.set("")
            self.score_combo.config(state="disabled")
            preview_lines.append("当前为多数据集模式：评分表将按各自数据集目录自动解析。")
            for dataset_name in dataset_names:
                score_files = _list_score_files(self.project_root, dataset_name)
                if score_files:
                    preview_lines.append(f"{dataset_name}: {score_files}")
                else:
                    preview_lines.append(f"{dataset_name}: 未发现 .xlsx，将自动进入无监督模式。")

        self._set_score_preview(preview_lines)

    def _set_running(self, running: bool):
        self.run_btn.config(state="disabled" if running else "normal")
        self.dataset_listbox.config(state="disabled" if running else "normal")
        self.select_all_btn.config(state="disabled" if running else "normal")
        self.clear_selection_btn.config(state="disabled" if running else "normal")
        self.refresh_btn.config(state="disabled" if running else "normal")
        self.refresh_scores_btn.config(state="disabled" if running else "normal")
        if running:
            self.score_combo.config(state="disabled")
            self._update_progress_ui(0.0, "任务已开始，等待处理...")
        else:
            self._refresh_score_files()
            if self.progress_var.get() >= 100.0:
                self.progress_status_var.set("已完成")
            else:
                self.progress_status_var.set("就绪")

    def _on_run_clicked(self):
        dataset_names = self._get_selected_dataset_names()
        score_file_name = self.score_file_var.get().strip() or None
        if not dataset_names:
            messagebox.showerror("错误", "请先选择至少一个数据集。")
            return

        self._set_running(True)
        self._update_progress_ui(0.0, f"准备运行 {len(dataset_names)} 个数据集...")
        self._append_log("=" * 72)
        self._append_log(f"[*] 开始运行全流程，共 {len(dataset_names)} 个数据集...")
        self._append_log(f"[*] 数据集列表：{', '.join(dataset_names)}")

        def worker():
            try:
                run_pipelines(
                    project_root=self.project_root,
                    cfg=self.cfg,
                    dataset_names=dataset_names,
                    score_file_name=score_file_name if len(dataset_names) == 1 else None,
                    n_splits=int(self.n_splits_var.get()),
                    overwrite=bool(self.overwrite_var.get()),
                    visualize=bool(self.visualize_var.get()),
                    reuse_existing=bool(self.reuse_existing_var.get()),
                    log=lambda s: self.root.after(0, self._append_log, s),
                    progress_callback=lambda v, s: self.root.after(0, self._update_progress_ui, v, s),
                )
                self.root.after(0, lambda: messagebox.showinfo("完成", f"全流程运行完成，共处理 {len(dataset_names)} 个数据集。"))
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
    dataset_names = args.dataset_name
    if not dataset_names:
        dataset_name = input("Enter dataset folder under data/: ").strip()
        dataset_names = [dataset_name] if dataset_name else []
    if not dataset_names:
        raise ValueError("dataset name cannot be empty.")
    if args.score_file and len(dataset_names) != 1:
        raise ValueError("--score-file 仅支持单数据集模式。")

    run_pipelines(
        project_root=project_root,
        cfg=cfg,
        dataset_names=dataset_names,
        score_file_name=args.score_file,
        n_splits=args.n_splits,
        overwrite=args.overwrite,
        visualize=args.visualize,
        reuse_existing=args.reuse_extracted,
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
    parser.add_argument("--dataset-name", nargs="+", default=None, help="One or more dataset folders under data/.")
    parser.add_argument("--score-file", default=None, help="Score matrix filename under dataset folder (single dataset only).")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for stratified CV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw feature files.")
    parser.add_argument("--visualize", action="store_true", help="Save feature-sequence plots during extraction.")
    parser.add_argument("--reuse-extracted", action="store_true", help="Reuse existing extracted feature CSVs when available.")
    args = parser.parse_args()

    if args.cli:
        _run_cli(args)
    else:
        _run_gui(args.config)


if __name__ == "__main__":
    main()
