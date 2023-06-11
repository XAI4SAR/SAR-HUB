# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
#from .swin_mlp import SwinMLP
from .tiny_vit import TinyViT

def build_model(mode,model_type):
    
    if mode == 'train':
        if model_type == 'Swin-T-BEN':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=19, ## 51 from million AID
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
        elif model_type == 'Swin-T-IMG':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=1000, ## 51 from million AID
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
        elif model_type == 'Swin-T-OSU':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-T-OPT':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=51, ## 51 from million AID
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
        elif model_type == 'Swin-T-TSX+OSU':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=42, ## 51 from million AID
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
        elif model_type == 'Swin-T-TSX':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=32, ## 51 from million AID
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
        elif model_type == 'Swin-B-OPT':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=45, ## 51 from million AID
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
        elif model_type == 'Swin-B-IMG':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=1000, ## 51 from million AID
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
        elif model_type == 'Swin-B-BEN':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=19, ## 51 from million AID
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
        elif model_type == 'Swin-B-TSX':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=32, ## 51 from million AID
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
        elif model_type == 'Swin-B-OSU':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
    elif mode == 'test':
        if model_type == 'Swin-B-TSX':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-B-BEN':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-B-OPT':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-T-OPT':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-T-TSX':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        elif model_type == 'Swin-T-BEN':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 51 from million AID
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
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")
    elif mode == 'transfer':
        if model_type == 'tiny_vit':
            model = TinyViT(img_size=128,
                            in_chans=3,
                            num_classes=10,   ## 10 from opensarurban
                            embed_dims=[96,192,384,576],
                            depths=[2,2,6,2],
                            num_heads=[3,6,12,18],
                            window_sizes=[4,4,8,4],
                            mlp_ratio=4.,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            use_checkpoint=False,
                            mbconv_expand_ratio=4.0,
                            local_conv_size=3,
                            layer_lr_decay=1.0,
                            )
        elif model_type == 'Swin':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=10, ## 10 from opensarurban
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
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")
    else:
        raise NotImplementedError(f"Unkown mode: {mode}")
    return model
