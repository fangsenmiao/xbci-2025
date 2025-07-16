import numpy as np
import matplotlib.pyplot as plt
from npy_folder_loader import loadnpyfolder

# 加载数据
root = 'data/T4'
total_data = loadnpyfolder(root)
data = total_data['data']
labels = total_data['label']

print("data shape:", data.shape)
print("labels shape:", labels.shape)

# 创建时间轴 (4秒，1992个采样点)
time = np.linspace(0, 4, 1992)

# 定义8种颜色对应8个通道
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

# 遍历所有数据样例
for i in range(len(data)):
    # 创建新图形
    plt.figure(figsize=(12, 8))
    plt.title(f'8-Channel Waveform - Sample {i + 1}, Label: {labels[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 绘制8个通道的数据
    for channel in range(8):
        #print(data[i][:, channel])
        plt.plot(time, data[i][:, channel],
                 color=colors[channel],
                 alpha=0.7,
                 label=f'Channel {channel + 1}')

    # 添加图例和网格
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    plt.tight_layout()
    plt.show()