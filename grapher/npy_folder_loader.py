import os
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor

def loadnpyfolder(folder_path):
    """加载文件夹中所有npy文件并返回数据和标签字典
    返回格式:
    {
        'data': numpy数组 (n_samples, 1992, 8),
        'label': numpy数组 (n_samples, 1)
    }
    """
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
                if eeg_data.shape == (1992, 8):
                    eeg_data = eeg_data.T  # 转置为(8, 1992)
                elif eeg_data.shape != (8, 1992):
                    # 处理形状不匹配的情况
                    if eeg_data.shape[0] == 8 and eeg_data.shape[1] > 1992:
                        eeg_data = eeg_data[:, :1992]
                    elif eeg_data.shape[1] == 8 and eeg_data.shape[0] > 1992:
                        eeg_data = eeg_data[:1992, :].T
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

                # 转置为(1992, 8)并添加到列表
                data_list.append(eeg_data.T)  # 现在形状是(1992, 8)
                label_list.append(label)

            except ValueError:
                print(f"警告: 无法解析标签在文件 {file_name}")

    if not data_list:
        raise ValueError(f"在文件夹 {folder_path} 中没有有效数据")

    # 堆叠为3维数组(n_samples, 1992, 8)
    data_array = np.stack(data_list, axis=0)  # 形状(n, 1992, 8)
    label_array = np.array(label_list).reshape(-1, 1)  # 形状(n, 1)

    return {'data': data_array, 'label': label_array}