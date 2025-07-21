"""
基于Transformer的EEG分类模型 - 支持5类别分类
修改版本：读取8通道1992采样点的EEG数据，支持5类别标签
数据格式：mi_<subject_id>_<label>_<index>.npy
"""

import os
import numpy as np
import math
import random
import time
import glob
import copy

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
from npy_folder_loader import loadnpyfolder
from datetime import datetime

# 设置CUDA环境变量
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                    nn.Dropout(0.7),
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
    def __init__(self, pretrained_path=None):
        super(Trans, self).__init__()
        # 训练参数
        self.batch_size = 50
        self.n_epochs = 1000
        self.img_height = 8  # EEG通道数 (修改为8)
        self.img_width = 1992  # 时间点数量 (修改为1992)
        self.channels = 1
        self.c_dim = 5  # 分类类别数 - 5类
        self.lr = 0.00005  # 学习率
        self.weight_decay = 1e-4  # 添加权重衰减
        self.b1 = 0.5  # Adam参数
        self.b2 = 0.9
        #self.nSub = nsub  # 受试者编号
        self.start_epoch = 0
        self.root = 'data/'  # 数据路径

        # CUDA张量类型
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        # 损失函数
        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()  # 分类损失

        # 初始化模型
        self.model = ViT(n_classes=5, in_channels=25).cuda()  # 5类
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 25, 1992))  # 打印模型结构 (修改为8通道1992采样点)

        self.centers = {}

        if pretrained_path:
            print(f"Loading pretrained model from {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.model.state_dict()

            # 只加载匹配的层参数
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and "clshead" not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)


        # 创建日志文件
        self.log_write = open(f"./results/log_unified_{timestamp}.txt", "w")
        print(f"初始化统一模型，支持5类分类")

    def get_source_data(self):

        # to get the data of target subject
        self.total_data = loadnpyfolder(self.root + 'T')
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']
        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        # test data
        # to get the data of target subject
        self.test_tmp = loadnpyfolder(self.root + 'E')
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))
        Wb = csp(tmp_alldata, self.allLabel-1)  # common spatial pattern
        self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)
        return self.allData, self.allLabel, self.testData, self.testLabel, target_mean, target_std, Wb

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"更新学习率为: {lr}")

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label

    def save_model(self, best_model_state, bestAcc, target_mean, target_std, Wb):
        """保存模型和预处理参数到同一文件"""
        folder_name = f"{timestamp}_{bestAcc:.4f}"
        model_dir = os.path.join('./', folder_name)
        os.makedirs(model_dir, exist_ok=True)

        # 创建包含模型状态和预处理参数的字典
        checkpoint = {
            'model_state_dict': best_model_state,
            'preprocess_params': {
                'target_mean': target_mean,
                'target_std': target_std,
                'Wb': Wb
            }
        }

        # 保存到单一文件
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(checkpoint, model_path)
        print(f"模型和参数已保存至: {model_dir}")

    def augment_data(self, x):
        """EEG数据增强方法"""
        # 添加高斯噪声
        noise = torch.randn_like(x) * random.uniform(0.01, 0.05)
        x_noised = x + noise

        # 随机时间偏移
        shift = random.randint(-10, 10)
        if shift > 0:
            x_shifted = torch.cat([x[..., shift:], x[..., :shift]], dim=-1)
        else:
            x_shifted = x

        # 随机通道置零（模拟电极接触不良）
        zero_mask = torch.ones_like(x)
        channels_to_zero = random.sample(range(self.img_height), k=random.randint(1, 2))
        zero_mask[:, :, channels_to_zero, :] = 0

        return x_noised * zero_mask, x_shifted * zero_mask

    def train(self):


        img, label, test_data, test_label, target_mean, target_std, Wb = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)



        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = torch.tensor([], dtype=torch.long).cuda()
        Y_pred = torch.tensor([], dtype=torch.long).cuda()


        # 添加最佳模型状态跟踪
        best_model_state = None
        best_val_loss = float('inf')
        patience, patience_counter = 10, 0  # 早停机制

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
            weight_decay=self.weight_decay
        )

        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.

        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()

            if e < 5:  # 前5个epoch使用更低的学习率
                self.update_lr(self.optimizer, self.lr * 0.1)
            else:
                self.update_lr(self.optimizer, self.lr)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {e}")
                break

            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # === 添加以下增强代码 ===
                aug_img1, aug_img2 = self.augment_data(img)
                #img = torch.cat([img, aug_img1, aug_img2], dim=0)
                #label = torch.cat([label, label, label], dim=0)
                # =====================

                # 使用增强后的数据和标签
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(
                    label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    # 保存当前最佳模型状态
                    best_model_state = copy.deepcopy(self.model.module.state_dict())
                    Y_true = test_label
                    Y_pred = y_pred

                # 每个epoch后在验证集上评估
                with torch.no_grad():
                    self.model.eval()
                    Tok, Cls = self.model(test_data)
                    val_loss = self.criterion_cls(Cls, test_label)

                    # 更新最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = copy.deepcopy(self.model.module.state_dict())
                    else:
                        patience_counter += 1


        if best_model_state is not None:
            self.save_model(best_model_state, bestAcc, target_mean, target_std, Wb)
        else:
            # 如果没有达到更好的准确率，保存最终模型
            self.save_model(self.model.module.state_dict(), bestAcc, target_mean, target_std, Wb)
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    result_write = open(f"./results/sub_result_{timestamp}.txt", "w")

    seed_n = np.random.randint(500)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    #print('Subject %d' % (i+1))
    #trans = Trans(i + 1)




    trans = Trans(pretrained_path='20250717_104427_0.6923/model.pth')
    bestAcc, averAcc, Y_true, Y_pred = trans.train()
    print('THE BEST ACCURACY IS ' + str(bestAcc))
    result_write.write('Subject ' + 'Seed is: ' + str(seed_n) + "\n")
    result_write.write('**Subject ' + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
    result_write.write('Subject ' + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
    # plot_confusion_matrix(Y_true, Y_pred, i+1)
    best = best + bestAcc
    aver = aver + averAcc
    yt = Y_true
    yp = Y_pred



    #best = best / 3
    #aver = aver / 3
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()






if __name__ == "__main__":
    main()
