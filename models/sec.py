import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class cnnRnn(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet18()
        encoder = nn.ModuleList(list(encoder.children())[:-2])
        self.encoder = nn.Sequential(*encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv3d(
            512, 512, stride=(2, 1, 1), kernel_size=(2, 3, 3), padding=(0, 1, 1)
        )  # hidden input
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output
        self.GeM = GeM()
        self.fc = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

    def forward(self, batch):
        batch = batch["images"]
        b, n, c, _, _ = batch.shape
        batch = rearrange(batch, "b n c h w -> (b n) c h w")
        x = self.encoder(batch)
        # rx = self.GeM(x)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=b, n=n)
        H = x[:, 0]
        inputs = x[:, 1:]
        H = nn.functional.relu(self.conv2(H))
        H = nn.functional.relu(self.fc2(self.GeM(H).view((b, -1))))
        Y = None
        for i in range(1, inputs.shape[1]):
            X = inputs[:, i]
            X_1 = inputs[:, i - 1]
            X_ = torch.cat((X.unsqueeze(1), X_1.unsqueeze(1)), dim=1)
            X_ = rearrange(X_, "b d c h w -> b c d h w")
            X_ = self.GeM(nn.functional.relu(self.conv1(X_)).squeeze(2))
            X_ = self.fc3(X_.view((b, -1)))
            H = nn.functional.relu(H + X_)
            Y = nn.functional.relu(self.fc3(H))
        Y = self.fc(Y)
        return Y


class sec(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet18(pretrained=True)
        encoder = nn.ModuleList(list(encoder.children())[:-3])
        self.encoder = nn.Sequential(*encoder)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.GeM = GeM()
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        # self.rnn = nn.RNN(512, 512, batch_first=True, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(512, 256)

    def forward(self, batch):
        batch = batch["images"]
        b, n, c, _, _ = batch.shape
        batch = rearrange(batch, "b n c h w -> (b n) c h w")
        x = self.encoder(batch)
        # rx = self.GeM(x)
        rx = rearrange(x, "(b n) c h w -> b c n h w", b=b, n=n)
        rx = self.pool(rx)
        rx = self.fc1(rx.view((b, -1)))

        # return rx[:, 0]
        return rx
        rx = self.rnn(rx)[0][:, -1]
        # print(rx.shape)
        # x = rearrange(x, "(b n) c h w -> b c n h w", b=b, n=n)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)
        x = self.fc1(rx)
        # print(x.shape)
        return x


class fpn(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet18(pretrained=True)
        encoder = nn.ModuleList(list(encoder.children())[:-2])
        self.conv1x1 = nn.ModuleDict()
        self.conv1x1_ = nn.ModuleDict()
        for i in range(3):
            self.conv1x1[str(i)] = nn.Conv2d(256, 256, kernel_size=1)
            self.conv1x1_[str(i)] = nn.Conv2d(256, 256, kernel_size=1)

        self.encoder = nn.Sequential(*encoder)
        self.fc = nn.Linear(256, 256)
        self.GeM = GeM()

    def forward(self, batch):
        batch = batch["images"]
        b, n, c, _, _ = batch.shape
        feature_map = {}
        batch = rearrange(batch, "b n c h w -> (b n) c h w")
        x = self.encoder[0](batch)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        feature_map["0"] = x
        for i in range(3):
            x = self.encoder[i + 4](x)
            feature_map[str(i + 1)] = x
        x = rearrange(x, "(b n) c h w -> b n c h w", b=b, n=n)
        x_ = self.conv1x1_["0"](x[:, 2]) + self.conv1x1["1"](x[:, 1])
        x = self.conv1x1_["1"](x_) + self.conv1x1["2"](x[:, 0])

        rx = self.GeM(x)
        # rx = rearrange(rx, "(b n) c h w -> b n c h w", b=b, n=n)
        x = self.fc(torch.flatten(rx, 1))
        return x


class fpn2(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet18(pretrained=False)
        encoder = nn.ModuleList(list(encoder.children())[:-2])
        self.conv1x1 = nn.ModuleList([nn.ModuleDict() for _ in range(4)])
        self.conv1x1_ = nn.ModuleList([nn.ModuleDict() for _ in range(4)])
        layers = [64, 64, 128, 256]
        for i in range(4):
            for j in range(3):
                self.conv1x1[i][str(j)] = nn.Conv2d(layers[i], layers[i], kernel_size=1)
                self.conv1x1_[i][str(j)] = nn.Conv2d(
                    layers[i], layers[i], kernel_size=1
                )
        self.fh_conv1x1 = nn.ModuleDict()
        self.fh_tconvs = nn.ModuleDict()
        for i in range(3, -1, -1):
            self.fh_conv1x1[str(i)] = nn.Conv2d(
                in_channels=layers[i], out_channels=256, kernel_size=1
            )
            self.fh_tconvs[str(i)] = torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            )

        self.encoder = nn.Sequential(*encoder)
        self.fc = nn.Linear(256, 256)
        self.GeM = GeM()

    def forward(self, batch):
        batch = batch["images"]
        b, n, c, _, _ = batch.shape
        feature_map = {}
        batch = rearrange(batch, "b n c h w -> (b n) c h w")
        x = self.encoder[0](batch)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        tmp = x
        x = rearrange(x, "(b n) c h w -> b n c h w", b=b, n=n)
        x_ = self.conv1x1_[0]["0"](x[:, 2]) + self.conv1x1[0]["1"](x[:, 1])
        x = self.conv1x1_[0]["1"](x_) + self.conv1x1[0]["2"](x[:, 0])
        feature_map["0"] = x
        for i in range(3):
            tmp = self.encoder[i + 4](tmp)
            x = tmp
            x = rearrange(x, "(b n) c h w -> b n c h w", b=b, n=n)
            x_ = self.conv1x1_[i + 1]["0"](x[:, 2]) + self.conv1x1[i + 1]["1"](x[:, 1])
            x = self.conv1x1_[i + 1]["1"](x_) + self.conv1x1[i + 1]["2"](x[:, 0])
            feature_map[str(i + 1)] = x
        # for i in feature_map:
        #     print(feature_map[i].shape)
        x = self.fh_conv1x1["3"](feature_map["3"])
        for i in range(3, 2, -1):
            x = self.fh_tconvs[str(i)](x)
            # print(i)
            # print(x.shape)
            x = x + self.fh_conv1x1[str(i - 1)](feature_map[str(i - 1)])

        rx = self.GeM(x)
        # rx = rearrange(rx, "(b n) c h w -> b n c h w", b=b, n=n)
        x = self.fc(torch.flatten(rx, 1))
        return x
