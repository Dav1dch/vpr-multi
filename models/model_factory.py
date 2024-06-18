# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.minkloc_multimodal import MinkLocMultimodal, ResnetFPN
from models.fpn3d import FPN3d
from models.ViT import ViTSec
from models.sec import fpn, sec, cnnRnn, fpn2
from misc.utils import MinkLocParams
import torch


def model_factory(params: MinkLocParams):
    in_channels = 1

    # MinkLocMultimodal is our baseline MinkLoc++ model producing 256 dimensional descriptor where
    # each modality produces 128 dimensional descriptor
    # MinkLocRGB and MinkLoc3D are single-modality versions producing 256 dimensional descriptor
    if params.model_params.model == "MinkLocRGB":
        image_fe_size = 256
        image_fe = ResnetFPN(
            out_channels=image_fe_size,
            lateral_dim=image_fe_size,
            fh_num_bottom_up=4,
            fh_num_top_down=0,
        )
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "fpn3d":
        image_fe_size = 256
        image_fe = FPN3d()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "sec":
        image_fe_size = 256
        image_fe = sec()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )

    elif params.model_params.model == "fpn":
        image_fe_size = 256
        image_fe = fpn2()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "vit":
        image_fe_size = 256
        image_fe = ViTSec(
            num_classes=1000,
            dim=1000,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.05,
            emb_dropout=0.05,
        )
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    else:
        raise NotImplementedError(
            "Model not implemented: {}".format(params.model_params.model)
        )

    return model
