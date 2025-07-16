import os
import numpy as np
import glob

def loadnpyfolder(folder_path):
    """加载文件夹中所有npy文件并返回数据和标签字典"""
    file_list = glob.glob(os.path.join(folder_path, "*.npy"))

    if not file_list:
        raise ValueError(f"在文件夹 {folder_path} 中未找到任何npy文件")

    data_list = []
    label_list = []

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')

        if len(parts) >= 4:
            try:
                label = int(parts[2])
                if label < 0 or label > 4:
                    print(f"警告: 无效标签 {label} 在文件 {file_name}")
                    continue

                eeg_data = np.load(file_path)

                # 确保数据形状为(8, 1992)
                if eeg_data.shape != (8, 1992):
                    if eeg_data.shape[1] > 1992:
                        eeg_data = eeg_data[:, :1992]
                    else:
                        # 其他情况用零填充
                        padding = np.zeros((8, 1992))
                        last_valid_values = eeg_data[:, -1]
                        min_dim0 = min(8, eeg_data.shape[0])
                        min_dim1 = min(1992, eeg_data.shape[1])
                        padding[:min_dim0, :min_dim1] = eeg_data[:min_dim0, :min_dim1]
                        for ch in range(8):
                            if min_dim1 < 1992:  # 如果数据长度不足
                                padding[ch, min_dim1:] = last_valid_values[ch]
                        eeg_data = padding

                # 不再转置，保留原始形状 (8, 1992)
                data_list.append(eeg_data)  # 直接添加 (8, 1992) 数组
                label_list.append(label + 1)

            except ValueError:
                print(f"警告: 无法解析标签在文件 {file_name}")

    if not data_list:
        raise ValueError(f"在文件夹 {folder_path} 中没有有效数据")

    # 修改点1：沿第0轴堆叠，形状变为 (n_trials, 8, 1992)
    data_array = np.stack(data_list, axis=0)  # 新形状: (trials, channels, samples)

    # 修改点2：标签直接转为列向量 (n_trials, 1)
    label_array = np.array(label_list).reshape(-1, 1)  # 形状: (trials, 1)

    return {'data': data_array, 'label': label_array}