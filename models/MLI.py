import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math

from utils.l2_norm import l2_norm

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MLI(nn.Module):

    def __init__(self, args=None):

        super().__init__()

        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.loss = self.args.loss
        self.resolution = 25
        self.disturb_num = self.args.disturb_num
        self.short_cut_weight = self.args.short_cut_weight
        self.lamda = self.args.lamda
        if self.resnet:
            self.num_channel = 640
            self.num_channel2 = 160
            self.feature_extractor = ResNet.resnet12(drop=True)
            self.feature_size = 640
            self.conv_block3 = nn.Sequential(
                BasicConv(self.num_channel // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.conv_block4 = nn.Sequential(
                BasicConv(self.num_channel, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.max3 = nn.AdaptiveMaxPool2d((1, 1))
            self.max4 = nn.AdaptiveMaxPool2d((1, 1))
            self.both_mlp2 = nn.Sequential(
                nn.BatchNorm1d(self.num_channel * self.disturb_num),
                nn.Linear(self.num_channel * self.disturb_num, self.num_channel * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp3 = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.layer2_branch = nn.Sequential(
                nn.Conv2d(self.num_channel2, 40, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Conv2d(40, self.disturb_num, kernel_size=3, stride=2, padding=1)
            )
        else:
            self.num_channel = 64
            self.num_channel2 = 64

            self.feature_extractor = Conv_4.BackBone(self.num_channel)
            self.feature_size = 64 *5 *5
            self.avg = nn.AdaptiveAvgPool2d((5, 5))
            self.both_mlp2 = nn.Sequential(
                nn.Linear(self.num_channel * self.disturb_num, self.num_channel * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp3 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.layer2_branch = nn.Sequential(
                nn.Conv2d(self.num_channel2, 30, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, self.disturb_num, kernel_size=3, stride=2, padding=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def integration(self, layer1, layer2):

        batch_size = layer1.size(0)
        channel_num = layer1.size(1)
        disturb_num = layer2.size(1)
        layer1 = layer1.unsqueeze(2)
        layer2 = layer2.unsqueeze(1)

        sum_of_weight = layer2.view(batch_size, disturb_num, -1).sum(-1) + 0.00001
        vec = (layer1 * layer2).view(batch_size, channel_num, disturb_num, -1).sum(-1)
        vec = vec / sum_of_weight.unsqueeze(1)
        vec = vec.view(batch_size, channel_num*disturb_num)
        return vec

    def get_cosine_dist(self, inp, way, shot):

        f1,f2,f3,f4 = self.feature_extractor(inp)

        heat_map = self.layer2_branch(f2)
        layer2 = nn.Sigmoid()(heat_map)
        layer2_vec0 = self.integration(f4, layer2)

        layer2_vec = self.both_mlp2(layer2_vec0)
        layer2_vec = self.short_cut_weight * layer2_vec0 + (1 - self.short_cut_weight) * layer2_vec

        support_f2 = layer2_vec[:way * shot].view(way, shot, self.num_channel * self.disturb_num).mean(1)
        query_f2 = layer2_vec[way * shot:]
        cos_f2 = F.linear(l2_norm(query_f2), l2_norm(support_f2))

        if self.resnet:
            f3 = self.conv_block3(f3)
            f4 = self.conv_block4(f4)
            f3 = self.max3(f3)
            f3 = f3.view(f3.size(0), -1)
            f4 = self.max4(f4)
            f4 = f4.view(f4.size(0), -1)
        else:
            f3 = self.avg(f3)
            f3 = f3.view(f3.size(0), -1)
            f4 = f4.view(f4.size(0), -1)

        f3 = (1 - self.short_cut_weight) * self.both_mlp3(f3) + self.short_cut_weight * f3
        support_f3 = f3[:way * shot].view(way, shot, -1).mean(1)
        query_f3 = f3[way * shot:]

        support_f3 = l2_norm(support_f3)
        query_f3 = l2_norm(query_f3)
        cos_f3 = F.linear(query_f3, support_f3)

        f4 = (1 - self.short_cut_weight) * self.both_mlp4(f4) + self.short_cut_weight * f4
        support_f4 = f4[:way * shot].view(way, shot, -1).mean(1)
        query_f4 = f4[way * shot:]

        support_f4 = l2_norm(support_f4)
        query_f4 = l2_norm(query_f4)
        cos_f4 = F.linear(query_f4, support_f4)
        return cos_f3,cos_f4,cos_f2


    def meta_test(self, inp, way, shot):

        cos_f3,cos_f4,cos_f2 = self.get_cosine_dist(inp=inp, way=way, shot=shot)
        scores = cos_f3+cos_f4+cos_f2

        _, max_index = torch.max(scores, 1)
        return max_index

    def forward(self, inp):

        cos_f3,cos_f4,cos_f2 = self.get_cosine_dist(inp=inp, way=self.way, shot=self.shots[0])

        return cos_f3,cos_f4,cos_f2

