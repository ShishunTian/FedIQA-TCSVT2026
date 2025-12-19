import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from flcore.trainmodel.HyperIQA.HyperNet import resnet50_backbone

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


class BaseHeadSplit_TReS(nn.Module):
    def __init__(self, backbone, head):
        super(BaseHeadSplit_TReS, self).__init__()
        self.base = backbone  # 前半部分
        self.head = head  # 后半部分

    def forward(self, x):
        # 使用 return_features=True 获取中间层特征
        out_t_o, layer4_o, consistloss = self.base(x)
        output, consistloss = self.head(out_t_o, layer4_o, consistloss)  # 后半部分处理
        return output, consistloss


# 定义前半部分模型，负责生成out_t_o和layer4_o
class TReSBackbone(nn.Module):
    def __init__(self, model):
        super(TReSBackbone, self).__init__()
        self.model = model

    def forward(self, x):
        out_t_o, layer4_o, consistloss = self.model(x, return_features=True)

        return out_t_o, layer4_o, consistloss


# 定义后半部分，负责最终拼接和预测
class TReSHead(nn.Module):
    def __init__(self, model):
        super(TReSHead, self).__init__()
        self.fc2 = model.fc2
        self.fc = model.fc

    def forward(self, out_t_o, layer4_o, consistloss):
        out_t_o = self.fc2(out_t_o)
        predictionQA = self.fc(torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))

        return predictionQA, consistloss

class HyperBackbone(nn.Module):
    def __init__(self):
        super(HyperBackbone, self).__init__()
        # 共享特征提取部分（例如 ResNet Backbone）
        self.res = resnet50_backbone(16, 224, pretrained=True)

    def forward(self, x):
        # 输出 target_in_vec 和 hyper_in_feat 作为特征
        res_out = self.res(x)
        return res_out# 只返回输入向量


class HyperHead(nn.Module):
    def __init__(self):
        super(HyperHead, self).__init__()
        # lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size
        # 16,                   112,                224,            112,            56,              28,                 14,         7
        self.hyperInChn = 112
        self.target_in_size = 224
        self.f1 = 112
        self.f2 = 56
        self.f3 = 28
        self.f4 = 14
        self.feature_size = 7
        feature_size = 7


        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, res_out):
        feature_size = self.feature_size
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out

