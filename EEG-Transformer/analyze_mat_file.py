import scipy.io
import numpy as np


def read_mat_file(file_path):
    """
    读取MAT文件并输出变量信息和维度，改进对复杂对象的处理
    """
    data = scipy.io.loadmat(file_path)

    print(f"\n{'=' * 50}")
    print(f"文件 '{file_path}' 包含以下变量:")
    print('=' * 50)

    for key in data:
        if key.startswith('__') and key.endswith('__'):
            continue

        value = data[key]

        print(f"\n变量名: {key}")
        print(f"数据类型: {type(value).__name__}")

        if isinstance(value, np.ndarray):
            print(f"维度: {value.shape}")
            print(f"数组大小: {value.size}")

            # 处理可能的MATLAB元胞数组（对象数组）
            if value.dtype == object:
                print("检测到对象数组(可能是MATLAB元胞数组)")
                display_object_array(value)
            else:
                display_array_contents(value)

    print('\n' + '=' * 50)
    print("文件读取完成")
    print('=' * 50)


def display_array_contents(arr):
    """显示数组内容，包括更详细的类型信息"""
    print(f"数据类型: {arr.dtype}")

    # 如果是数值数组
    if np.issubdtype(arr.dtype, np.number):
        print(f"数值范围: [{arr.min():.6g} ~ {arr.max():.6g}]")
        print("前3x3数据示例:")

        # 根据数组维度决定如何切片
        if arr.ndim == 1:
            print(arr[:min(3, arr.size)])
        elif arr.ndim == 2:
            print(arr[:min(3, arr.shape[0]), :min(3, arr.shape[1])])
        else:
            # 多维数组只显示第一个切片
            slices = [0] * arr.ndim
            slices[0] = slice(0, min(3, arr.shape[0]))
            slices[1] = slice(0, min(3, arr.shape[1]))
            print(arr[tuple(slices)])

    # 如果是结构数组
    elif arr.dtype.names is not None:
        print("检测到结构数组")
        print(f"字段名: {arr.dtype.names}")
        if arr.size > 0:
            for i, name in enumerate(arr.dtype.names):
                print(f"  {name}:", end=' ')
                field = arr[0][name]

                if isinstance(field, np.ndarray):
                    print(f"数组 {field.shape} ({field.dtype})")
                else:
                    print(f"{type(field).__name__}")

            print("\n第一个元素的数据示例:")
            for name in arr.dtype.names:
                print(f"  {name}: {arr[0][name]}")

    # 如果是字符数组或字符串
    elif arr.dtype.kind in {'U', 'S'}:
        print("字符串内容:")
        if arr.size == 1:
            try:
                # 尝试提取实际的字符串
                if arr.dtype.kind == 'S':
                    s = arr.item().decode('utf-8', errors='replace')
                else:
                    s = arr.item()
                print(f"  {s}")
            except:
                print(f"  {arr}")
        else:
            # 处理字符串数组
            try:
                flat_arr = arr.ravel()
                for i, s in enumerate(flat_arr[:min(5, flat_arr.size)]):
                    if arr.dtype.kind == 'S':
                        s = s.decode('utf-8', errors='replace')
                    print(f"  {i}: {s}")
                if arr.size > 5:
                    print(f"  显示前5/{arr.size}个元素...")
            except:
                print(f"  {arr}")


def display_object_array(obj_arr):
    """处理MATLAB元胞数组（对象数组）"""
    print(f"对象数组内容 ({obj_arr.size}个元素):")
    flat_arr = obj_arr.ravel()

    for i, item in enumerate(flat_arr[:min(5, flat_arr.size)]):
        print(f"\n元素 {i}:")

        if isinstance(item, np.ndarray):
            # 递归显示数组信息
            display_array_contents(item)
        elif 'scipy.sparse' in str(type(item)):
            print(f"稀疏矩阵: {type(item).__name__}")
            print(f"  维度: {item.shape}")
            print(f"  非零元素: {item.nnz}")
        else:
            print(f"  类型: {type(item).__name__}")
            print(f"  值: {item}")

    if obj_arr.size > 5:
        print(f"\n显示前5/{obj_arr.size}个元素...")

if __name__ == "__main__":
    # 用户输入文件路径
    file_path = 'D:/MI/BCICIV_2a_mat/A01E.mat'
    #file_path = 'D:/MI/true_labels/A01E.mat'
    print(scipy.io.loadmat(file_path)['data'].shape)
    try:
        read_mat_file(file_path)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查: 1) 文件路径是否正确 2) 文件是否为有效的MAT格式")