import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
        model_name:str='dinov2_vitb14',
        num_trainable_blocks:int=2  ,
        norm_layer:bool=True,
        return_token:bool=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token


    def forward(self, x: torch.Tensor, flag: bool):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        # x: torch.Tensor = x["images"]
        # x = x.squeeze(1)
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        # f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        f = f.reshape((B, 1, -1, self.num_channels)).permute(0, 3, 1, 2)
        x = self.model.head(x)

        if self.return_token:
            return f, t
        return f, None

class SeqDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = DINOv2()
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)
        self.net_vlad = NetVLADLoupe(feature_size=768, max_samples=int(640*2), cluster_size=64,  # before 11.12 --- 64
                                     output_dim=256, gating=True, add_batch_norm=False,   # output_dim=512
                                     is_training=True)

    def forward(self, x, flag):
        x: torch.Tensor = x["images"]
        out_l_seq= None
        for i in range(x.shape[1]):
            seq: torch.Tensor = x[:, i]
            seq, _ = self.dino(seq, flag)
            if out_l_seq == None:
                out_l_seq = seq
            else:
                out_l_seq = torch.cat((out_l_seq, seq), dim=-1)
        out_l_seq = out_l_seq.squeeze(2)
        out_l_seq = out_l_seq.permute(2, 0, 1)
        out_l_seq = self.transformer_encoder2(out_l_seq)
        out_l_seq = out_l_seq.permute(1, 2, 0)
        out_l_seq = out_l_seq.unsqueeze(3)
        out_l_seq = self.net_vlad(out_l_seq)
        # out_l_seq = F.normalize(out_l_seq, dim=1)
        return out_l_seq, None




def test():
    model = DINOv2()
    model = model.cuda()
    print(model.model)
    img = {"images" : torch.rand((1, 1, 3, 420, 420)).cuda()}
    print(model(img, None).shape)
    print(model.model(img["images"].squeeze(1)).shape)
    return

def testseq():
    model = SeqDino().cuda()
    img = {"images": torch.rand((1, 5, 3, 224, 224)).cuda()}
    model(img, False)



if __name__ == "__main__":
    testseq()
