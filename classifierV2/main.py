"""
基于Transformer的EEG分类模型 - 支持5类别分类
修改版本：读取8通道1992采样点的EEG数据，支持5类别标签
数据格式：mi_<subject_id>_<label>_<index>.npy
"""

import argparse
import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob

import pandas as pd

import torch
import torch.nn.functional as F


from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (8, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(5080, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=5, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, input_pth, model_pth):
        super(ExP, self).__init__()
        self.batch_size = 30
        self.n_epochs = 2000
        self.img_height = 8  # EEG通道数 (修改为8)
        self.c_dim = 5
        self.lr = 0.00005
        self.weight_decay = 1e-4  # 添加权重衰减
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        #self.nSub = nsub

        self.start_epoch = 0
        self.root = input_pth

        #self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model, self.target_mean, self.target_std = self.load_model_and_preprocess(model_pth + 'model.pth')
        # summary(self.model, (1, 8, 1992))

    def load_test_npyfolder(self, folder_path):
        """加载测试文件夹中所有npy文件，返回文件名列表和数据数组"""
        file_list = glob.glob(os.path.join(folder_path, "*.npy"))

        if not file_list:
            raise ValueError(f"在文件夹 {folder_path} 中未找到任何npy文件")

        name_list = []  # 存储文件名的列表
        data_list = []  # 存储数据数组的列表

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            try:
                eeg_data = np.load(file_path)

                # 数据预处理
                if eeg_data.shape != (8, 1992):
                    if eeg_data.shape[1] > 1992:
                        eeg_data = eeg_data[:, :1992]
                    else:
                        padding = np.zeros((8, 1992))
                        last_valid_values = eeg_data[:, -1]
                        min_dim0 = min(8, eeg_data.shape[0])
                        min_dim1 = min(1992, eeg_data.shape[1])
                        padding[:min_dim0, :min_dim1] = eeg_data[:min_dim0, :min_dim1]
                        for ch in range(8):
                            if min_dim1 < 1992:
                                padding[ch, min_dim1:] = last_valid_values[ch]
                        eeg_data = padding

                name_list.append(file_name)
                data_list.append(eeg_data.transpose())

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {str(e)}")
                continue

        if not data_list:
            raise ValueError(f"在文件夹 {folder_path} 中没有有效数据")

        # 转换为数组
        name_array = np.array(name_list)  # 文件名数组
        data_array = np.stack(data_list, axis=2)  # 三维数据数组 (8, 1992, n_samples)

        return name_array, data_array

    def get_source_data(self):

        # to get the data of target subject
        self.file_list, self.test_data = self.load_test_npyfolder(self.root)

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)

        self.testData = self.test_data

        # standardize
        self.testData = (self.testData - self.target_mean) / self.target_std


        return self.testData

    def load_model_and_preprocess(self, model_path):

        checkpoint = torch.load(model_path, weights_only=False)

        # 1. 加载预处理参数
        preprocess_data = checkpoint['preprocess_params']
        target_mean = preprocess_data['target_mean']
        target_std = preprocess_data['target_std']

        # 2. 加载模型
        model = Conformer(n_classes=5)  # 根据你的模型定义创建实例
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 设置为评估模式

        return model, target_mean, target_std


    def classfy(self):
        # 初始化模型 (需与训练时结构一致)
        test_data = self.get_source_data()
        #print(test_data.shape)
        self.model = self.model.cuda()

        with torch.no_grad():
            tensor = torch.tensor(test_data).float().cuda()
            _, outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)

        #print(f"预测类别: {predicted_class[36]}")
        #print(f"各类别概率: {probabilities.cpu().numpy()}")
        return predicted_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/acc-test', help='测试文件夹路径')
    parser.add_argument('--output', type=str, default='results.csv', help='结果输出路径')
    parser.add_argument('--model', type=str, default='model/', help='模型文件地址')
    args = parser.parse_args()

    exp = ExP(args.input, args.model)
    predicted_class = exp.classfy()

    #print(predicted_class)
    tensor_cpu = predicted_class.cpu()

    file_list = exp.file_list
    predicted_array = tensor_cpu.numpy()
    df = pd.DataFrame({"file": file_list, "result": predicted_array})
    df.to_csv(args.output, index=False)  # 不保存行索引


if __name__ == "__main__":
    main()
