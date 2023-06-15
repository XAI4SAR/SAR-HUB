# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
#from .swin_mlp import SwinMLP
from .tiny_vit import TinyViT

def build_model(model_type,pre_class):
        if model_type == 'Swin-T':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=pre_class, ## 51 from million AID
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=4,    #default:7
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)
        elif model_type == 'Swin-B':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=pre_class, ## 51 from million AID
                                    embed_dim=128,
                                    depths=[2, 2, 18, 2],
                                    num_heads=[4, 8, 16, 32],
                                    window_size=4,    #default:7
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")
        return model
