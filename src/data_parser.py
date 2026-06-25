import os
import re


def parse_label(filename):
    """获取文件名中最后一个数字序列，作为打分标签"""
    base = os.path.splitext(filename)[0]
    intensity_match = re.search("\u529b\u5ea6\\s*([123])", base)
    if intensity_match:
        # Existing plots use numeric labels 1/3/5 for classes A/B/C.
        return {1: 1, 2: 3, 3: 5}[int(intensity_match.group(1))]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return None


def parse_suffix_type(filename):
    """根据文件名后缀判断数据集子集类型，返回"A"、"B"、"1"或None"""
    base = os.path.splitext(filename)[0]
    if "\u72ec\u5531\u6a21\u5f0f" in base:
        return "A"
    if "\u5408\u5531\u6a21\u5f0f" in base:
        return "B"
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
    return None


def parse_pitch_digit(filename):
    """从文件名中提取音高信息，返回数字部分（如4），如果没有匹配则返回None"""
    base = os.path.splitext(filename)[0]
    match = re.search(r"([A-Ga-g])(\d)", base)
    if match:
        return int(match.group(2))
    return None
