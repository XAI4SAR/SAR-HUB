import intial_multi_gpu
import dist_train
def config_train():
    
    config = {} 
    config['if_sto'] = 1
    config['if_save'] = 1
    config['train_batch_size'] = 128
    config['val_batch_size'] = 200
    config['num_epochs'] = 300
    config['models'] = {'save_model_path': ' ',     # 'Path to Save SAR Pre-trained Weights'
                        'load_model_path': ' '      # 'Path to load Pre-trained Weights'
        }
    config['Model_Type'] = 'ResNet-50'     # Model Type: ResNet-18,ResNet-50,ResNet-101,
    config['loss_type'] = 'Mini_CB_FL'   #! CE:cross entropy；FL：Focal Loss；CB_CE：Class-Balanced-CE；CB_FL：Class-Balanced-FL；Mini_CB_CE：Mini-Class-Balanced-CE；Mini_CB_FL：Mini-Class-Balanced-FL
    config['dataset'] = 'TerraSAR-X' #! TerraSAR-X、BigEarthNet-Small or OpenSARUrban
    
    
    if config['dataset'] == 'TerraSAR-X':
        config['datatxt_train'] = 'data/tsx_train.txt'
        config['datatxt_val'] = 'data/tsx_val.txt'
        config['nor_mean'] = 0.17721633016340846
        config['nor_std'] = 0.023696591996910408
        config['cate_num'] = 32
        
    elif config['dataset'] == 'BigEarthNet-Small':
        config['datatxt_train'] = 'data/ben_train.txt'
        config['datatxt_val'] = 'data/ben_val.txt'
        config['nor_mean'] = 0.5995
        config['nor_std'] = 0.0005743462
        config['cate_num'] = 19
        
    elif config['dataset'] == 'OpenSARUrban':
        config['datatxt_train'] = 'data/osu_train.txt'
        config['datatxt_val'] = 'data/osu_val.txt'
        config['nor_mean'] = 0.3159206957121415
        config['nor_std'] = 0.034312685984107194
        config['cate_num'] = 10
        
    return config

if __name__ == '__main__':
    config = config_train()
    world_size = intial_multi_gpu.init_distributed_mode()
    dist_train.train(config, world_size)