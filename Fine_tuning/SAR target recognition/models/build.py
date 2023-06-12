# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer

def build_model(mode,model_type):
    
    if mode == 'train':
        if model_type == 'Swin-T-BEN':
            model = SwinTransformer(img_size=128,
                                    patch_size=4, 
                                    in_chans=3,
                                    num_classes=19, 
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
                                    num_classes=1000, 
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
                                    num_classes=10, 
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
                                    num_classes=51, 
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
                                    num_classes=42, 
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
                                    num_classes=32, 
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
                                    num_classes=45, 
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
                                    num_classes=1000, 
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
                                    num_classes=19, 
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
                                    num_classes=32, 
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
                                    num_classes=10, 
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
    else:
        raise NotImplementedError(f"Unkown mode: {mode}")
    return model
