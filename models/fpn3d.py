import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.resnet3d import resnet18, resnet10
from einops.layers.torch import Rearrange
from torchvision.models._api import Weights, WeightsEnum
from functools import partial

from dgl.nn.pytorch import SAGEConv
import dgl


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        self.transform = nn.Sequential(Rearrange("b d c h w -> b c d h w"))

        r3d = models.video.r3d_18(pretrained=True)
        r3d = nn.ModuleList(list(r3d.children())[:-3])
        self.encoder = r3d
        for param in self.encoder.parameters():
            param.requires_grad = False
        # self.conv3d_1 = nn.Sequential(
        #     nn.Conv3d(64, 256, kernel_size=(3, 1, 1)),
        #     Rearrange("b c d h w -> b (d c) h w"),
        # )
        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1)),
            Rearrange("b c d h w -> b (d c) h w"),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        self.conv3d_3 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1)),
            Rearrange("b c d h w -> b (d c) h w"),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        # self.conv3d_2t = nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # )

        # self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.GeM = GeM()
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        x = x["images"]
        # x = x
        # Bottom-up using backbone
        x = self.transform(x)
        x = self.encoder[0](x)

        x = self.encoder[1](x)
        # print(self.pool(x).shape)
        # feature_map_1 = self.conv3d_1(x)

        x = self.encoder[2](x)
        feature_map_2 = self.conv3d_2(x)
        x = self.encoder[3](x)
        x = self.conv3d_3(x) + feature_map_2
        # feature_map = F.adaptive_avg_pool2d(x, (5, 5))
        x = self.fc(torch.flatten(self.GeM(x), 1))
        return x, None


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, f):
        h = self.linear1(f)
        h = F.leaky_relu(h)
        h = self.linear2(h)
        h = F.leaky_relu(h)
        h = self.linear3(h)
        h = F.leaky_relu(h)
        h = self.linear4(h)
        # h = F.relu(h)
        h = torch.sigmoid(h)
        # h = F.relu(h)
        # return torch.sigmoid(h)
        return h

class myGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNN, self).__init__()

        self.MLP = MLP(in_feats, 1)
        self.BN = nn.BatchNorm1d(in_feats)
        self.conv1 = SAGEConv(in_feats, in_feats, "mean")
        # self.conv2 = SAGEConv(in_feats, in_feats, "mean")

    def apply_edges(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        score = self.MLP(h_u - h_v)
        return {"score": score}

    def forward(self, g, x):
        # x = self.BN(x)

        with g.local_scope():
            g.ndata["x"] = x
            g.apply_edges(self.apply_edges)
            e = g.edata["score"]

        A = self.conv1(g, x, e)
        A = F.leaky_relu(A)
        # A = self.conv2(g, A, e)
        # A = F.leaky_relu(A)
        A = F.normalize(A, dim=1)
        # pred2, A2 = self.conv2(g, pred)
        return A, e



class FPN3d(nn.Module):

    def __init__(self):
        super(FPN3d, self).__init__()
        self.transform = nn.Sequential(Rearrange("b d c h w -> b c d h w"))

        r3d = models.video.r3d_18(pretrained=True)
        r3d = nn.ModuleList(list(r3d.children())[:-3])
        self.encoder = r3d
        for param in self.encoder.parameters():
            param.requires_grad = False
        # self.conv3d_1 = nn.Sequential(
        #     nn.Conv3d(64, 256, kernel_size=(3, 1, 1)),
        #     Rearrange("b c d h w -> b (d c) h w"),
        # )
        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1)),
            Rearrange("b c d h w -> b (d c) h w"),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        self.conv3d_3 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1)),
            Rearrange("b c d h w -> b (d c) h w"),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        # self.conv3d_2t = nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # )

        # self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.GeM = GeM()
        self.fc = nn.Linear(256, 256)
        self.fusion = nn.Conv3d(256, 256, kernel_size=(2, 1, 1))

        src = np.array([list(range(9)) for _ in range(9)]).reshape((-1,))
        dst = np.repeat(np.array(list(range(9))), 9)

        g = dgl.graph((src, dst))
        self.g = g.to("cuda")

        self.gnn = myGNN(256, 256, 128)

    def forward(self, x, flag=True):
        x = x["images"]
        # x = x
        # Bottom-up using backbone
        x = self.transform(x)
        x = self.encoder[0](x)

        x = self.encoder[1](x)
        # print(self.pool(x).shape)
        # feature_map_1 = self.conv3d_1(x)

        x = self.encoder[2](x)
        feature_map_2 = self.conv3d_2(x)
        x = self.encoder[3](x)
        feature_map_3 = self.conv3d_3(x)
        x = self.fusion(torch.stack((feature_map_2, feature_map_3), dim=2)).squeeze(2)
        # feature_map = F.adaptive_avg_pool2d(x, (5, 5))
        x = self.fc(torch.flatten(self.GeM(x), 1))

        if flag:
            x, _ = self.gnn(self.g, x)
        return x, None
