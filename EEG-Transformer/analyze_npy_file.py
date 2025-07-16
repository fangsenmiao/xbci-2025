import numpy as np
import os
import argparse

def analyze_npy_file(file_path, max_elements=200000):
    """分析.npy文件并输出其内容和维度信息"""
    try:
        # 加载.npy文件
        file_path = "d:\Desktop\EEG-Transformer-main\data\mi_01_0_13.npy"
        data = np.load(file_path)
        
        # 获取文件基本信息
        file_size = os.path.getsize(file_path)
        
        # 输出文件信息
        print(f"\n{'='*50}")
        print(f"文件路径: {file_path}")
        print(f"文件大小: {file_size/1024:.2f} KB ({file_size} 字节)")
        print(f"数据类型: {data.dtype}")
        print(f"数组维度: {data.shape} (共 {data.ndim} 维)")
        print(f"元素总数: {data.size}")
        
        # 输出数组内容
        print("\n数组内容:")
        if data.size <= max_elements:
            # 如果元素较少，打印全部内容
            print(data)
        else:
            # 对于大型数组，打印部分内容
            print("(数组较大，仅显示部分内容)")
            
            # 打印前几个元素
            if data.ndim == 1:
                print(f"前5个元素: {data[:5]}")
                print(f"后5个元素: {data[-5:]}")
            elif data.ndim == 2:
                print(f"左上角 3x3 子矩阵:")
                print(data[:3, :3])
                print(f"右下角 3x3 子矩阵:")
                print(data[-3:, -3:])
            else:
                print("前5个元素:")
                print(data.flat[:5])
                print("后5个元素:")
                print(data.flat[-5:])
        
        # 输出统计信息
        if np.issubdtype(data.dtype, np.number):
            print("\n统计信息:")
            print(f"最小值: {np.min(data)}")
            print(f"最大值: {np.max(data)}")
            print(f"平均值: {np.mean(data):.4f}")
            print(f"标准差: {np.std(data):.4f}")
        
        print('='*50)
        
    except Exception as e:
        print(f"\n处理文件 {file_path} 时出错: {str(e)}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='分析.npy文件内容和维度')
    argspath = "d:\Desktop\EEG-Transformer-main\data\mi_01_0_15.npy"
    # 处理单个文件
    argsmax = 20000
    if os.path.isfile(argspath) and argspath.endswith('.npy'):
        analyze_npy_file(argspath, argsmax)
    
    # 处理目录
    elif os.path.isdir(argspath):
        print(f"\n扫描目录: {argspath}")
        npy_files = [f for f in os.listdir(argspath) if f.endswith('.npy')]
        
        if not npy_files:
            print("目录中没有找到.npy文件")
            return
            
        print(f"找到 {len(npy_files)} 个.npy文件")
        
        for filename in npy_files:
            file_path = os.path.join(argspath, filename)
            analyze_npy_file(file_path, argsmax)
    
    else:
        print(f"错误: '{argspath}' 不是有效的.npy文件或目录")

if __name__ == "__main__":
    main()


