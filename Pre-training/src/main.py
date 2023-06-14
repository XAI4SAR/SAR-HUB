import intial_multi_gpu
import dist_train
import argparse
def config_train(args):
    
    config = {} 
    config['if_sto'] = args.if_sto
    config['train_batch_size'] = args.train_batch
    config['val_batch_size'] = args.val_batch
    config['num_epochs'] = args.num_epochs
    config['models'] = {'save_model_path': args.save_model_path,    
                        'load_model_path': args.pretrained_path      
        }
    config['Model_Type'] = args.model     
    config['loss_type'] = args.loss   
    config['dataset'] = args.dataset 
    config['DRAE'] = args.DRAE
    
    if config['dataset'] == 'TerraSAR-X':
        if config['DRAE'] == 'Reinhard':
            config['datatxt_train'] = 'data/tsx_train.txt'
            config['datatxt_val'] = 'data/tsx_val.txt'
            config['nor_mean'] = 0.17721633016340846
            config['nor_std'] = 0.023696591996910408
            config['cate_num'] = 32
            config['para'] = [3.5,4.5]
        else:
            raise NameError('Non-corresponding DRAE functions and dataset.')
        
    elif config['dataset'] == 'BigEarthNet-Small':
        if config['DRAE'] == 'PTLS':
            config['datatxt_train'] = 'data/ben_train.txt'
            config['datatxt_val'] = 'data/ben_val.txt'
            config['nor_mean'] = 0.5995
            config['nor_std'] = 0.0005743462
            config['cate_num'] = 19
            config['para'] = [0,2]
        else:
            raise NameError('Non-corresponding DRAE functions and dataset.')
        
    elif config['dataset'] == 'OpenSARUrban':
        if config['DRAE'] == 'PTLS':
            config['datatxt_train'] = 'data/osu_train.txt'
            config['datatxt_val'] = 'data/osu_val.txt'
            config['nor_mean'] = 0.3159206957121415
            config['nor_std'] = 0.034312685984107194
            config['cate_num'] = 10
            config['para'] = [0,3]
        else:
            raise NameError('Non-corres ponding DRAE functions and dataset.')
        
    return config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='Upstream_Training')
    
    parser.add_argument('--pretrained_path', default='resnet18_I_nwpu_cate45.pth', help='Path of the optical remote sensing pre-trained backbone path.')
    parser.add_argument('--save_model_path', type=str, default='model/ResNet18/', help='Path of where to save the SAR pre-trained models.')
    parser.add_argument('--if_sto', type=int, default=1, help='I')
    parser.add_argument('--model', type=str, default='ResNet18', help='The model type in training process.')
    # Optional: ResNet18 ResNet50 DenseNet121 SENet50 MobileV3 Swin-T Swin-B
    parser.add_argument('--dataset', type=str, default='TerraSAR-X', help='The dataset in training process.')
    # Optional: TerraSAR-X BigEarthNet-Small OpenSARUrban
    parser.add_argument('--train_batch', type=int, default=128, help='The instances number of each batch during training.')
    parser.add_argument('--val_batch', type=int, default=200, help='The instances number of each batch during validation.')
    parser.add_argument('--num_epochs', type=int, default=300, help='The number of training epochs.')
    parser.add_argument('--loss',default='CB_Focal', help='The loss in training procedure.')
    # Optional: CB_Focal Mini_CB_FL
    parser.add_argument('--DRAE', type=str, default='PTLS', help='The DRAE function used in the training.')
    # PTLS Reinhard
    
    args = parser.parse_args([])
    
    config = config_train(args)
    world_size = intial_multi_gpu.init_distributed_mode()
    dist_train.train(config, world_size)