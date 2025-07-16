"""_summary_
本代码处理16进制原始EEG信号数据（适用于1,2,4,8通道）。
原始信号的每行(帧)数据格式："时间,帧头+eeg_data+帧尾"
输出结果为每个marker开始后4秒数据的npy文件，形状为[8,2000]
"""

import json
import numpy as np
import os

# 常量定义
BYTES_PER_SAMPLE = 3  # 每个采样点占3字节
BYTES_PER_LINE = 192  # 每行纯数据192字节
HEX_CHARS_PER_BYTE = 2  # 每个字节用2个十六进制字符表示
EEG_SCALE_FACTOR = (4.5 / 8388607) / 8 * 1e6  # 转换为微伏的缩放因子
THRESHOLD_VALUE = int("0x800000", 16)  # 阈值，用于判断是否为负数
MAX_INT_24_BIT = 16777216  # 2^24，用于将补码转换为有符号整数
SAMPLE_RATE = 500  # 采样率(Hz)
SECONDS_PER_SEGMENT = 4  # 每个标记后需要的秒数
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SECONDS_PER_SEGMENT  # 每个片段的采样点数


def one_ch_one_sampling_process(data):
    """用来处理单通道1次采样的三字节数据

    Args:
        data (str): 三字节数据 (DA43F2)

    Returns:
        int: 经过解析后的单通道1次采样的微伏数据
    """
    # 将16进制字符串转换为整数
    eeg_hex = int(data, 16)

    # 检查是否超过阈值，如果超过则为负数，需要转换
    if eeg_hex < THRESHOLD_VALUE:
        eeg_raw = eeg_hex
    else:
        eeg_raw = eeg_hex - MAX_INT_24_BIT

    # 应用缩放因子转换为微伏单位
    eeg_microvolts = eeg_raw * EEG_SCALE_FACTOR
    return int(eeg_microvolts)


def get_eeg_signal(input_path, channel_num):
    """获取eeg原始16进制数据，解析成10进制数，并解析marker

    Args:
        input_path (str): 数据地址
        channel_num (int): 通道数

    Raises:
        ValueError: 输入channel_num错误
        ValueError: 输入数据行长度错误

    Returns:
        dict: EEG解析后dict
    """

    # 验证channel_num合法性
    if channel_num not in {1, 2, 4, 8}:
        raise ValueError("channel_num must be 1, 2, 4 or 8")

    # 初始化结果字典，为每个通道创建空列表
    final_dict = {f"C{i}": [] for i in range(1, channel_num + 1)}
    final_dict["stim"] = []  # 保存标记数据: [[marker1, idx1], [marker2, idx2], ...]

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()  # 格式: 时间戳,AB04C00C0101......DD
                if not line:  # 跳过空行
                    continue

                # 分割时间戳和数据部分
                parts = line.split(",")
                if len(parts) != 2:
                    print(f"警告: 第{line_num + 1}行格式错误，跳过此线")
                    continue

                one_frame_data = parts[1]  # 带帧头帧尾的EEG数据

                # 验证行长度
                expected_length = BYTES_PER_LINE * HEX_CHARS_PER_BYTE + 12
                if len(one_frame_data) != expected_length:
                    raise ValueError(
                        f"行 {line_num + 1} 数据错误: 应为 {expected_length} 字符，实际为 {len(one_frame_data)}")

                # 解析marker
                if one_frame_data[8:10] != "01":
                    marker = marker_query.get(one_frame_data[8:10], "UNKNOWN")
                    # 计算当前标记在数据中的索引位置
                    sample_idx = len(final_dict["C1"])  # 使用当前数据长度作为索引
                    final_dict["stim"].append([marker, sample_idx])

                # 提取有效数据部分（去掉帧头和帧尾）
                valid_data = one_frame_data[10:-2]

                # 计算每行包含的样本块数
                samples_per_channel = BYTES_PER_LINE // (BYTES_PER_SAMPLE * channel_num)

                # 遍历每个样本块
                for block in range(samples_per_channel):
                    start_pos = block * BYTES_PER_SAMPLE * channel_num * HEX_CHARS_PER_BYTE

                    # 提取每个通道的6字符(3字节)十六进制数据
                    for ch in range(channel_num):
                        chunk_start = start_pos + ch * BYTES_PER_SAMPLE * HEX_CHARS_PER_BYTE
                        chunk_end = chunk_start + BYTES_PER_SAMPLE * HEX_CHARS_PER_BYTE
                        hex_str = valid_data[chunk_start:chunk_end]

                        # 解析单个通道单次采样数据
                        uv_value = one_ch_one_sampling_process(hex_str)

                        # 保存到结果字典
                        final_dict[f"C{ch + 1}"].append(uv_value)

        return final_dict
    except FileNotFoundError:
        print(f"错误: 文件 {input_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 处理文件时发生异常: {e}")
        return None


def extract_and_save_segments(eeg_dict, output_dir, channel_num=8):
    """从EEG数据中提取每个marker后的4秒数据并保存为npy文件

    Args:
        eeg_dict (dict): EEG数据字典，包含各通道数据和标记
        output_dir (str): 输出目录
        channel_num (int): 通道数，默认为8
    """
    if not eeg_dict:
        print("没有EEG数据可处理")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有标记
    markers = eeg_dict["stim"]

    # 提取各通道数据
    channel_data = [eeg_dict[f"C{i + 1}"] for i in range(channel_num)]

    # 检查数据长度是否一致
    data_length = len(channel_data[0])
    for ch_data in channel_data[1:]:
        if len(ch_data) != data_length:
            print("错误: 各通道数据长度不一致")
            return

    # 处理每个标记
    for i, (marker, start_idx) in enumerate(markers):
        end_idx = start_idx + SAMPLES_PER_SEGMENT

        # 检查片段是否在数据范围内
        if end_idx > data_length:
            print(f"警告: 标记 {marker} (索引 {start_idx}) 的4秒数据超出范围，只有 {data_length - start_idx} 个样本可用")
            continue

        # 提取4秒数据，形状为 [channel_num, SAMPLES_PER_SEGMENT]
        segment = np.array([ch_data[start_idx:end_idx] for ch_data in channel_data])

        # 保存为npy文件
        output_file = os.path.join(output_dir, f"segment_{marker}_{i}.npy")
        np.save(output_file, segment)
        print(f"已保存 {marker} 的第 {i} 个片段到 {output_file}，形状: {segment.shape}")


if __name__ == "__main__":
    # 初始化标记查询字典
    """
    详情见 “maker协议.xlsx”
    """
    marker_query = {}
    # 数字1~9对应"11"~"19"
    for i in range(1, 10):
        key = f"{10 + i}"  # "11"到"19"
        marker_query[key] = str(i)
    # 字母A~Z对应"20"~"45"
    for i in range(26):
        key = f"{20 + i}"
        value = chr(ord('A') + i)  # A到Z
        marker_query[key] = value
    # 字母a~z对应"46"~"71"
    for i in range(26):
        key = f"{46 + i}"
        value = chr(ord('a') + i)  # a到z
        marker_query[key] = value

    # 需要修改的部分
    """
    在这里修改参数
    """
    file_path = r"C:\JXJ_8-BCI\8-BCI20250708161342_EEG_test_男.txt"
    channel_num = 8
    output_dir = os.path.dirname(file_path) + "/segments"  # 输出目录

    # 解析EEG数据
    eeg_dict = get_eeg_signal(file_path, channel_num)

    # 提取并保存每个标记后的4秒数据
    if eeg_dict:
        extract_and_save_segments(eeg_dict, output_dir, channel_num)