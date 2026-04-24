# acoustics_score_analysis

本项目用于对声乐技巧评分（1/3/5）进行建模与解释分析，基于音频提取的声学参数进行序数回归训练，并提供特征组合对比与统计关联分析。

## 项目目的

- 使用可解释的声学特征对声乐技巧评分进行预测；
- 比较不同特征组合的分类效果，找到表现最优的组合；
- 输出统计分析结果，识别对评分影响显著的声学参数。

## 工作流程

1. **特征提取**
   - 在 `Extract/` 中运行相应脚本，生成特征 CSV 文件
   - 每个脚本会在数据目录下创建对应的输出文件夹
   - 特征包括：Jitter、Shimmer（Praat shimmer local）、H1H2_output、HNR、Q1、SpectralSlope、LowFreqEnergyRatio、HighFreqNoiseRatio、CPP 等
   - 其中 `Q1` 定义为第一共振峰品质因子（first-formant quality factor）：
     - `Q1 = F1 / BW1`
     - `F1` 为第一共振峰中心频率（Hz），`BW1` 为第一共振峰带宽（Hz）
   - `H1H2_output` 使用 output-based 计算：在 `F0` 与 `2F0` 附近小频带内搜索局部峰值，再计算 `H1-H2`
2. **特征组合训练与评估**
   - 运行 `TrainModels.py`
   - 自动枚举可用特征组组合，训练序数回归（LogisticAT）
   - 输出：
     - `特征组合对比结果.csv`
     - `模型对比结果.csv`
     - `分层10折交叉验证结果.csv`
     - 混淆矩阵与 PCA / t-SNE 可视化
3. **声学参数影响分析**
   - 运行 `OrdinalFeatureAnalysis.py`
   - 输出单因素与多因素序数回归结果、VIF、多重共线性与显著性分析
   - 输出：
     - `声学参数与评分关联分析结果.csv`
     - `单因素序数回归结果.csv`
     - `特征相关矩阵.csv`
     - `特征VIF结果.csv`

## 运行环境

- Python 3.x
- 主要依赖：numpy、pandas、scikit-learn、mord、statsmodels、librosa、matplotlib

## 目录说明

