import os
from typing import List, Optional

import pandas as pd

from src.data_parser import parse_suffix_type


class CombinedData:
    """
    封装合并后的声乐技巧评分与声学特征统计量数据。
    提供灵活的数据访问、子集过滤和列分离功能。
    """
    def __init__(self, df_score: pd.DataFrame, df_stats: pd.DataFrame):
        """
        初始化并合并数据。

        Args:
            df_score: 评分 DataFrame (索引为文件名，列为技巧名)。
            df_stats: 特征统计 DataFrame (索引为文件名，列为特征名)。
        """
        # 1. 确保索引名称一致且为 'audio_filename' (可选，但推荐)
        if df_score.index.name is None:
            df_score.index.name = 'audio_filename'
        if df_stats.index.name is None:
            df_stats.index.name = 'audio_filename'

        # 2. 执行 Inner Join (只保留两者都有的文件)
        self._full_df = pd.merge(
            df_score,
            df_stats,
            left_index=True,
            right_index=True,
            how='inner'
        ).dropna()

        if self._full_df.empty:
            raise ValueError("合并后的数据集为空！请检查评分文件和特征文件的文件名是否匹配。")

        # 去掉Jitter值最大的行
        max_jitter_idx = self._full_df['Jitter'].idxmax()
        self._full_df = self._full_df.drop(index=max_jitter_idx)

        # 3. 自动识别技巧列和特征列
        # 逻辑：前 N 列是技巧 (根据 df_score 的列数)，之后全是特征
        n_tech_cols = len(df_score.columns)
        all_cols = self._full_df.columns.tolist()

        self.tech_cols = all_cols[:n_tech_cols]
        self.feat_cols = all_cols[n_tech_cols:]

        print(f"[+] 数据集初始化完成:")
        print(f"  - 总样本数: {len(self._full_df)}")
        print(f"  - 技巧列 ({len(self.tech_cols)}): {self.tech_cols}")
        print(f"  - 特征列 ({len(self.feat_cols)}): {self.feat_cols}")

    @property
    def full_df(self) -> pd.DataFrame:
        """获取完整的合并 DataFrame (包含所有行和所有列)。"""
        return self._full_df

    def get_full_table(self) -> pd.DataFrame:
        """别名：获取整张大表 (audio_filename, techs..., feats...)。"""
        # 重置索引以便文件名作为第一列显示，方便查看或保存
        df = self._full_df.reset_index()
        return df

    def get_score_table(self) -> pd.DataFrame:
        """获取仅包含技巧评分的表 (audio_filename, techs...)。"""
        df = self._full_df[self.tech_cols].reset_index()
        return df

    def get_feat_table(self) -> pd.DataFrame:
        """获取仅包含特征统计的表 (audio_filename, feats...)。"""
        df = self._full_df[self.feat_cols].reset_index()
        return df

    def get_scores_subset(self, subset_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取仅包含技巧评分的表 (audio_filename, techs...) 的指定子集。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        df = self._filter_by_subset(self._full_df[self.tech_cols], subset_types)
        return df.reset_index()

    def get_one_score_subset(
            self,
            tech_name: str,
            subset_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取指定技巧和子集的表 (audio_filename, tech_name)。
        tech_name: 技巧列名 (如 'vibrato')。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        if tech_name not in self.tech_cols:
            raise ValueError(f"技巧 '{tech_name}' 不在已知技巧列表中：{self.tech_cols}")

        df = self._filter_by_subset(self._full_df[[tech_name]], subset_types)
        return df.reset_index()

    def get_feats_subset(self, subset_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取仅包含特征统计的表 (audio_filename, feats...) 的指定子集。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        df = self._filter_by_subset(self._full_df[self.feat_cols], subset_types)
        return df.reset_index()

    def get_scores_feats_subset(self, subset_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取指定子集的完整合并表 (audio_filename, techs..., feats...)。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        df = self._filter_by_subset(self._full_df, subset_types)
        return df.reset_index()

    def get_one_score_feats_subset(
            self,
            tech_name: str,
            subset_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取指定技巧和子集的表 (audio_filename, tech_name, feats...)。
        tech_name: 技巧列名 (如 'vibrato')。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        if tech_name not in self.tech_cols:
            raise ValueError(f"技巧 '{tech_name}' 不在已知技巧列表中：{self.tech_cols}")

        cols = [tech_name] + self.feat_cols
        df = self._filter_by_subset(self._full_df[cols], subset_types)
        return df.reset_index()

    def get_one_score_selected_feats_subset(
            self,
            tech_name: str,
            feat_names: List[str],
            subset_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取指定技巧、特征和子集的表 (audio_filename, tech_name, feat_names...)。
        tech_name: 技巧列名 (如 'vibrato')。
        feat_names: 特征列名列表 (如 ['Jitter', 'Shimmer'])。
        subset_types: 子集列表 (如 ['A', '1'])。若为 None，则返回全部。
        """
        if tech_name not in self.tech_cols:
            raise ValueError(f"技巧 '{tech_name}' 不在已知技巧列表中：{self.tech_cols}")
        for feat in feat_names:
            if feat not in self.feat_cols:
                raise ValueError(f"特征 '{feat}' 不在已知特征列表中：{self.feat_cols}")

        cols = [tech_name] + feat_names
        df = self._filter_by_subset(self._full_df[cols], subset_types)
        return df.reset_index()

    @staticmethod
    def _filter_by_subset(df: pd.DataFrame, subset_types: Optional[List[str]]) -> pd.DataFrame:
        """内部辅助函数：根据文件名后缀过滤行。"""
        if subset_types is None:
            return df.copy()

        # 使用 map 应用 parse_suffix_type
        mask = df.index.map(lambda x: parse_suffix_type(x) in subset_types)
        filtered_df = df[mask]

        if len(filtered_df) == 0:
            print(f"[!] 警告：筛选子集 {subset_types} 后无剩余数据。")

        return filtered_df

    def save_to_csv(self, output_dir: str, filename: str = "score_feats_data.csv"):
        """保存完整表格到 CSV。"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        self.get_full_table().to_csv(path, index=False)
        print(f"[+] 完整数据已保存至：{path}")


def combine_score_feats_data(
        df_score: pd.DataFrame,
        df_stats: pd.DataFrame,
        output_dir: str = None
) -> pd.DataFrame:
    """
    [兼容层] 构建并返回合并后的 DataFrame。
    内部使用 CombinedDataset 类处理逻辑。
    """
    dataset = CombinedData(df_score, df_stats)
    if output_dir:
        dataset.save_to_csv(output_dir)
    return dataset.get_full_table()
