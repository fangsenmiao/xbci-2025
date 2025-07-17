"""
基于Transformer的EEG分类模型 - 支持5类别分类
修改版本：读取8通道1992采样点的EEG数据，支持5类别标签
数据格式：mi_<subject_id>_<label>_<index>.npy
"""

import os
import numpy as np
import pandas as pd
import math
import random
import time
import glob
import argparse

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchsummary import summary

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp  # 导入CSP空间滤波器实现
from npy_test_folder_loader import load_test_npyfolder
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置CUDA环境变量
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    """将EEG信号转换为嵌入向量 - 适配8通道数据"""
    def __init__(self, emb_size):
        super().__init__()
        # 使用卷积层提取特征并重组为序列 - 修改为适应8通道数据
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),  # 时间维度卷积
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (25, 5), stride=(1, 5)),  # 空间维度卷积 - 输入8通道
            Rearrange('b e (h) (w) -> b (h w) e'),  # 重组为序列格式
        )
        # 分类令牌（CLS token）
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
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
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
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
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
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
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out

class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(1992),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class channel_attention(nn.Module):
    """通道注意力机制 - 适配8通道输入"""

    def __init__(self, sequence_num=1992, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)

        self.query = nn.Sequential(
            nn.Linear(25, 25),  # 保持输入输出维度相同
            nn.LayerNorm(25),
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(25, 25),
            nn.LayerNorm(25),
            nn.Dropout(0.3)
        )
        # 修改1: 移除投影层中的维度变化
        self.projection = nn.Sequential(
            nn.Linear(25, 25),  # 输出维度保持8
            nn.LayerNorm(25),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out

class Trans():
    """Transformer模型训练框架 - 修改为处理npy文件，支持5类"""
    def __init__(self, input_pth, model_pth):
        super(Trans, self).__init__()
        # 训练参数
        self.batch_size = 50
        self.n_epochs = 100
        self.img_height = 8  # EEG通道数 (修改为8)
        self.img_width = 1992  # 时间点数量 (修改为1992)
        self.channels = 1
        self.c_dim = 5  # 分类类别数 - 5类
        self.lr = 0.0002  # 学习率
        self.b1 = 0.5  # Adam参数
        self.b2 = 0.9
        #self.nSub = nsub  # 受试者编号
        self.start_epoch = 0
        self.root = input_pth  # 数据路径
        self.model_pth = model_pth  # 模型路径

        # CUDA张量类型
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        # 损失函数
        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()  # 分类损失

        # 初始化模型
        self.model, self.target_mean, self.target_std, self.Wb = self.load_model_and_preprocess(model_pth + 'model.pth', model_pth + 'preprocess_params.npz')



        self.centers = {}

        # 创建日志文件
        self.log_write = open(f"./results/log_unified_{timestamp}.txt", "w")
        #print(f"初始化统一模型，支持5类分类")

    def get_source_data(self):

        # to get the data of target subject
        self.file_list, self.test_data = load_test_npyfolder(self.root)

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)

        self.testData = self.test_data

        # standardize
        self.testData = (self.testData - self.target_mean) / self.target_std


        self.testData = np.einsum('abcd, ce -> abed', self.testData, self.Wb)
        return self.testData

    def load_model_and_preprocess(self, model_path, preprocess_path):
        # 1. 加载预处理参数
        preprocess_data = np.load(preprocess_path)
        target_mean = preprocess_data['target_mean']
        target_std = preprocess_data['target_std']
        Wb = preprocess_data['Wb']

        # 2. 加载模型
        model = ViT(n_classes=5)  # 根据你的模型定义创建实例
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置为评估模式

        return model, target_mean, target_std, Wb


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
    parser.add_argument('--input', type=str, default='data/T', help='测试文件夹路径')
    parser.add_argument('--output', type=str, default='results.csv', help='结果输出路径')
    parser.add_argument('--model', type=str, default='d:/Desktop/EEG-trainers/EEG-Transformer/20250716_172458_0.6154/', help='模型文件地址')
    args = parser.parse_args()

    trans = Trans(args.input, args.model)
    predicted_class = trans.classfy()

    #print(predicted_class)
    tensor_cpu = predicted_class.cpu()

    file_list = trans.file_list
    predicted_array = tensor_cpu.numpy()
    df = pd.DataFrame({"file": file_list, "result": predicted_array})
    df.to_csv(args.output, index=False)  # 不保存行索引


if __name__ == "__main__":
    main()
