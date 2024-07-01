# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: architecture of SeqOT


import torch
import torch.nn as nn
import math
# sys.path.append('../tools/')
# from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates

        return activation

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(
            cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad
class featureExtracter(nn.Module):
    def __init__(self, seqL=5):
        super(featureExtracter, self).__init__()

        self.seqL = seqL

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2,1), stride=(2,1), bias=False)
        self.conv1_add = nn.Conv2d(16, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)

        self.relu = nn.ReLU(inplace=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.convLast2 = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=int(900*self.seqL), cluster_size=64,  # before 11.12 --- 64
                                     output_dim=256, gating=True, add_batch_norm=False,   # output_dim=512
                                     is_training=True)


    def forward(self, x_l: torch.Tensor):
        out_l_seq = None
        for i in range(self.seqL):

            one_x_l_from_seq = x_l[:, i:(i+1), :, :]

            out_l = self.relu(self.conv1(one_x_l_from_seq))
            out_l = self.relu(self.conv1_add(out_l))
            out_l = self.relu(self.conv2(out_l))
            out_l = self.relu(self.conv3(out_l))
            out_l = self.relu(self.conv4(out_l))
            out_l = self.relu(self.conv5(out_l))
            out_l = self.relu(self.conv6(out_l))
            out_l = self.relu(self.conv7(out_l))


            out_l_1 = out_l.permute(0,1,3,2)
            out_l_1 = self.relu(self.convLast1(out_l_1))
            out_l = out_l_1.squeeze(3)
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)
            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            if i==0:
                out_l_seq = out_l
            else:
                print(out_l_seq.shape)
                out_l_seq: torch.Tensor = torch.cat((out_l_seq, out_l), dim=-2)

        out_l_seq = out_l_seq.squeeze(3)
        out_l_seq = out_l_seq.permute(2, 0, 1)
        out_l_seq = self.transformer_encoder2(out_l_seq)
        out_l_seq = out_l_seq.permute(1, 2, 0)
        out_l_seq = out_l_seq.unsqueeze(3)
        print(out_l_seq.shape)
        out_l_seq = self.net_vlad(out_l_seq)
        out_l_seq = F.normalize(out_l_seq, dim=1)
        print(out_l_seq.shape)
        return out_l_seq



if __name__ == '__main__':
    amodel = featureExtracter(5).cuda()
    test = torch.rand((1, 5, 32, 900)).cuda()
    print(amodel(test).shape)
