<div align="center">

# <img src="assets/logo.png" alt="SEA" width="5%"> &nbsp; 基于 Transformer 的时空特征学习实现的 EEG 信号分类器

![](https://github.com/LYZ2024/pictures/blob/main/9dc3ab91a92c898ce73c5569a4e22ce.png?raw=true)

## ❓ 简介

Classfier 主要使用 Python 编写，主要功能是利用训练好的模型对新的运动想象脑电数据进行分类，判断是哪一种运动信号。


## ⚡️ 参考代码

参考了 github 开源项目 EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization [here](https://github.com/eeyhsong/EEG-Conformer)

## ⚡️ 快速开始

1. **设置 Python 环境:** 

   ```shell
   conda create -n new_env python=3.10 -y
   conda activate new_env
   ```

2. **安装 eegcls 依赖项:** 
   ```shell
   cd classifierV2
   pip install -r requirements.txt
   ```

3. **进行信号分类:**

   默认为当前目录下的模型文件夹和当前目录下的测试样例，输出结果保存在当前路径下的results.csv中，也可以通过传入超参数改变上述位置。
   ```shell
   python main.py --input[测试文件夹路径] --output[结果输出路径] --model[模型文件地址]
   ```
   **_Tips: 模型文件地址如下：例如模型地址为‘d:/Desktop/classifierV2/model/model.pth’，命令中可以写为“--model d:/Desktop/classifierV2/model”或“classifierV2/model”_**

## 🔎 关于模型

模型为.pth格式文件，包含一个权重模型和若干用于标准化输入数据的相关参数，通过 EEG-Conformer 框架训练得到，[here](https://github.com/fangsenmiao/xbci-2025/tree/main)
