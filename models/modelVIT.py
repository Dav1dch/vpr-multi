import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torchvision.transforms as transforms


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, x2):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q, _, _ = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        _, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv2)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNormAttn(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, x2):
        for attn, ff in self.layers:
            x = attn(x, x2) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()

        num_patches = 60 * 80
        patch_height = 2
        patch_width = 2
        patch_dim = 256 * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim)
        )  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 256),
        )

    def forward(self, img1, img2):
        x = self.to_patch_embedding(
            img1
        )  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        x2 = self.to_patch_embedding(img2)
        b, n, _ = (
            x.shape
        )  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(
            self.cls_token, "() n d -> b n d", b=b
        )  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # 将cls_token拼接到patch token中去       (b, 65, dim)

        x2 = torch.cat((cls_tokens, x2), dim=1)
        x += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x2 += self.pos_embedding[:, : (n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)
        x2 = self.dropout(x2)

        x = self.transformer(x, x2)  # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # (b, dim)

        x = self.to_latent(x)  # Identity (b, dim)
        # print(x.shape)

        return self.mlp_head(x)  #  (b, num_classes)


def test():
    model = ViT(
        image_size=(480, 640),
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=3,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    ).cuda()

    img = torch.randn(16, 256, 60, 80).cuda()
    img2 = torch.randn(16, 256, 60, 80).cuda()
    preds = model(img, img2)
    print(preds.shape)
    return


if __name__ == "__main__":
    test()
