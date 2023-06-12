# SAR-HUB

## Introduction

This project is for paper "SAR-HUB: Pre-training, Fine-tuning, and Explaining".

### Features

1.  **Pre-training:** Deep neural networks are trained with large-scale, open-source SAR scene image datasets.
    
2.  **Fine-tuning:** The pre-trained DNNs are transferred to diverse SAR downstream tasks.
    
3.  **Explaining:** Benefits of SAR pre-trained models in comparison to optical pre-trained models are explained.
    

![The project overview.](img/intro.png)

We release this repository with reproducibility (open-source code and datasets), generalization (sufficient experiments on different tasks), and explainability (qualitative and quantitative explanations).

### Contributions

-   An optimization method for large-scale SAR image classification is proposed to improve model performance.
    
-   A novel explanation method is proposed to explain the benefits of SAR pre-trained models qualitatively and quantitatively.
    
-   The Model-Hub offers a variety of SAR pre-trained models validated on various SAR benchmark datasets.
    

## Previously on SAR-HUB

In our previous work, we discussed what, where, and how to transfer effectively in SAR image classification and proposed the SAR image pre-trained model (ResNet-18) based on large-scale SAR scene classification that achieved good performance in SAR target recognition downstream task. We tentatively analyzed the generality and specificity of features in different layers to demonstrate the advantage of SAR pre-trained models.

```LaTeX
@article{huang2019,
  title={What, where, and how to transfer in SAR target recognition based on deep CNNs},
  author={Huang, Zhongling and Pan, Zongxu and Lei, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={58},
  number={4},
  pages={2324--2336},
  year={2019},
  publisher={IEEE}
}
```

```LaTeX
@article{huang2020,
  title={Classification of large-scale high-resolution SAR images with deep transfer learning},
  author={Huang, Zhongling and Dumitru, Corneliu Octavian and Pan, Zongxu and Lei, Bin and Datcu, Mihai},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={18},
  number={1},
  pages={107--111},
  year={2020},
  publisher={IEEE}
}
```

Based on the preliminary findings in our previous work, we released this **SAR-HUB** project as a continuous study with the following extensions:

-   To further improve the large-scale SAR scene classification performance and the feature generalization ability, we propose an optimization method with **dynamic range adapted augmentation (DRAA)** and **mini-batch class imbalanced** **loss function** **(mini-CBL)**.
    
-   In pre-training, **7** popular CNN and Transformer based architectures and **3** different large-scale SAR scene image datasets are explored, collected in Model-Hub. In fine-tuning, **7** different SAR downstream tasks are evaluated.
    
-   We propose **SAR** **knowledge point (SAR-KP)** concept, together with CAM based methods, to explain why the SAR pre-trained models outperform ImageNet and optical remote sensing image pre-trained models in transfer learning.
    

## Getting Started

### Requirements

Please refer to [requirements](requirements) for installation.

If you need to conduct experiments of SAR scene classification, target recognition or SAR knowledge point, please download the required dependencies according to [here](XAI4SAR/SAR-HUB/requirements/scene_classification.txt).

If you need to conduct experiments of SAR object detection or sementic segmentation, please refer to [object_detection.txt](requirements/object_detection.txt) and [sementic_segmentation.txt](requirements/sementic_segmentation.txt) respectively. 

### Pre-training And Fine-tuning

#### **Scene Classification**

* Data Preparation

  Please download BigEarthNet-S1.0 datasets and OpenSARUrban datasets first from:

  BigEarthNet-S1.0: https://bigearth.net/

  OpenSARUrban: https://pan.baidu.com/s/1D2TzmUWePYHWtNhuHL7KdQ

  After that, please normalize the datasets to 0-1 and store them in *npy* format. The file directory tree is as below:

  ```
  ├── dataset
  │   ├── BigEarthNet
  │   │   ├── BEN-S1-npy
  │   ├── OpenSARUrban
  │   │   ├── OSU-npy
  ├── data
  │   ├── BEN
  │   │   ├── test.txt
  │   │   ├── train.txt
  │   │   ├── val.txt
  │   ├── OSU
  │   │   ├── test.txt
  │   │   ├── train.txt
  │   │   ├── val.txt
  ├──models
  │   ├── build.py
  │   ├── swin_load_pretrain.py
  │   ├── ...
  ├──src
  │   ├── main.py
  │   ├── test.py
  │   ├── ...
  ```

  Of course, if you want to store data in your own style, then please change the *137th* and *93rd* lines of [datasets.py](SAR_scene_classification/src/dataset.py) according to the data path you store.

* Train and Test

  We recommend you to use GPUs but not a CPU to train and test, because it will greatly shorten the time.

  Before starting training, please change the parameters in *main.py*, including the dataset, model, path and so on.

  Then you can use the command below to start a training procedure:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main.py > result.txt
  ```
  If you want to use a single GPU, set *CUDA_VISIBLE_DEVICES* to the serial number of a single GPU and change *--nproc_per_node* to 1. For example:
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 main.py > result.txt
  ```
  The results will be written to *result.txt* when using *nohup*. If you want to observe the training process on the terminal, delete *nohup* and *> result.txt*.

  After training, please change the parameters in *test.py* and use the command below to start test:

  ```bash
  python test.py
  ```

  The results will be given on the terminal.

  **The explanation of significant code file or folder is as follows**:

  - **main.py**: Code for significant parameters. The main parameters are needed checking in this file. You need to start traning from this file.

  - **intial_multi_gpu.py**: Code for intializing the multi-gpu process. In most cases, there is no need to change it.

  - **dist_train.py** and **dist_val.py**: Code for training and validation in each epoch. In most cases, there is no need to change it.

  - **test.py**: Code for test for the final models.

  - **dataset.py** and **read_dataset.py**: Code for reading data of each datasets. You may need to change them because we use the *npy* format in experiments.

  - **loss.py**: Code for several losses used.

  - **transform.py** and **data_transform.py**: Code for several basic transformation used in the experiments.

  - **model_prepare.py**: Configuration code for loading models.

  - **models**: Base configuration folder for Network structure code. In most cases, there is no need to change it.
  

#### **Target Recognition**

* Data Preparation

  We use MSTAR dataset, OpenSARShip dataset and FuSARShip dataset for SAR target recognition tasks. Therefore you may need to download their firstly.

  OpenSARShip: https://opensar.sjtu.edu.cn/

  FuSARShip: https://radars.ac.cn/web/data/getData?dataType=FUSAR

  MSTAR：https://pan.baidu.com/s/1SAdmYAOHPheAH98CLP9dQg Code: h2ig

  The file directory tree is as below:

  ```
  ├── dataset
  │   ├── MSTAR
  │   │   ├── SOC
  │   │   │   ├── train
  │   │   │   ├── ...
  │   ├── FuSARShip
  │   │   │   ├── train
  │   │   │   ├── ...
  │   ├── OpenSARShip
  │   │   │   ├── train
  │   │   │   ├── ...
  ├── data
  │   ├── FSS
  │   │   ├── test.txt
  │   │   ├── train.txt
  │   ├── MSTAR
  │   │   ├── test.txt
  │   │   ├── train_10.txt
  │   │   ├── train_30.txt
  │   │   ├── ...
  │   ├── OSS
  │   │   ├── ...
  ├──models
  │   ├── build.py
  │   ├── swin_load_pretrain.py
  │   ├── ...
  ├──src
  │   ├── main.py
  │   ├── test.py
  │   ├── ...
  ```

  * Train

  Before starting training, please change the parameters in *main.py*, including model type and so on.

  Then you can use the command below to start a training procedure:
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup python train.py > SENet_FuSARship_TSX.txt
  ```
  
  The results will be written to *SENet_FuSARship_TSX.txt* when using *nohup*. If you want to observe the training process on the terminal, delete *nohup* and *> SENet_FuSARship_TSX.txt*.

  **The explanation of significant code file or folder is as follows**:

  - **main.py**: Code for training and validation in each epoch. The main parameters are needed checking in this file. You need to start traning from this file.
 
  - **sampler.py**: Code for ImbalancedDatasetSampler.

  - **{}_dataset.py** and **read_dataset.py**: Code for reading data of each datasets. You may need to change them because we use the *npy* format in experiments.

  - **transform.py** and **data_transform.py**: Code for several basic transformation used in the experiments.

  - **network.py**: Configuration code for CNN network structure code. In most cases, there is no need to change it.

  - **models**: Base configuration folder for ViT structure code. In most cases, there is no need to change it.

#### **Object Detection**

The object detection are based on MMDetection framework,combining Feature Pyramid Networks (FPN) and Fully Convolutional One Stage (FCOS), and we have not changed any of it. Therefore, we only give the *SAR config* and *\_\_base\_\_* and introduce how to use them.

* Data Preparation

  You need to download the SSDD dataset, HRSID dataset and LS-SSDDv1.0 dataset for this task. 

  SSDD: https://pan.baidu.com/s/1sVs63jB_aM-RbcHEaWQgTg Code: 4pz1

  HRSID: https://aistudio.baidu.com/aistudio/datasetdetail/54512
  
  LS-SSDDv1.0: https://radars.ac.cn/web/data/getData?newsColumnId=6b535674-3ce6-43cc-a725-9723d9c7492c

  The file directory tree in MMDetection is as below:

  ```
  ├── mmdetection
  │   ├── configs
  │   │   ├── SAR
  │   │   │   ├── SAR config
  │   │   │   │   ├── fcos_r18_caffe_fpn_gn-head_4x4_HRSID.py
  │   │   │   │   ├── fcos_r50_caffe_fpn_gn-head_4x4_HRSID.py
  │   │   │   │   ├── ...
  │   │   │   ├── __base__
  │   │   │   │   ├── datasets
  │   │   │   │   │   ├── SSDD.py
  │   │   │   │   │   ├── HRSID.py
  │   │   │   │   │   ├── ...
  │   │   │   │   ├── SAR data
  │   │   │   │   │   ├── HRSID
  │   │   │   │   │   │   ├── train.json
  │   │   │   │   │   │   ├── test.json
  │   │   │   │   │   ├── ...
  │   │   │   │   ├── schedules
  │   │   │   │   │   ├── schedule_1x.py
  ```

  * Train

  The training procedure is the same as the official MMDetection's. You can use the command below to start a training procedure:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python tools/train.py mmdetection-master/configs/SAR/SAR config/fcos_r18_caffe_fpn_gn-head_4x4_HRSID.py
  ```
  
  The results will be written to the log save path you set in each config file.

#### **Sementic Segmentation**

We adopt DeepLabv3 under MMSegmentation framework during the experiments. Similar to the object detection task, we give the *SAR config* and *\_\_base\_\_* and introduce how to use them.

* Data Preparation

  You need to download the SpaceNet6 dataset for this task. 

  SpaceNet6: https://spacenet.ai/sn6-challenge/

  The file directory tree in MMSegmentation is as below:

  ```
  ├── MMSegmentation
  │   ├── configs
  │   │   ├── SAR
  │   │   │   ├── SAR config
  │   │   │   │   ├── deeplabv3_r18_20k_SN6.py
  │   │   │   │   ├── deeplabv3_r50_20k_SN6.py
  │   │   │   │   ├── ...
  │   │   │   ├── __base__
  │   │   │   │   ├── datasets
  │   │   │   │   │   ├── mvoc.py
  │   │   │   │   ├── SAR data
  │   │   │   │   │   ├── train.txt
  │   │   │   │   │   ├── test.txt
  │   │   │   │   ├── schedules
  │   │   │   │   │   ├── schedule_20k.py
  ```

  * Train

  The training procedure is the same as the official MMSegmentation's. You can use the command below to start a training procedure:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/SAR/SAR config/deeplabv3_d121_20k_SN6.py
  ```
  
  The results will be written to the log save path you set in each config file.

### Explaining

#### **Data preparation**

We use SAR KP on the MSTAR dataset. So you need to download it firstly:

MSTAR：https://pan.baidu.com/s/1SAdmYAOHPheAH98CLP9dQg Code: h2ig

The code are proposed [here](SAR_KP).

#### **Train and Get_KP**

The ResNet-50 model used in KP is firstly trained on the MSTAR dataset. Then we connect it with U-Net to explain it. Before training, you need to change the parameters in the *train.py*, including the save path and the loaded model.

You can use the command below to start the training:

```bash
  python train.py
```
Notably, we don't use validation dataset during train, so you may need to use *tensorboard* or other training process visualization tool to check whether the training is normal by observing the loss curve. 

After training, you need to run the *test_Get_KP.py* to get KP. The visualization and the values of disturbance will be saved in *jpg* and *npy* format respectively.

If you want to get more intuitive visualization results, you can use *KP_visual.py* to colored the disturbance.

#### **Explanation of significant code file or folder**

- **train.py**: Code for training. The main parameters are needed checking in this file. You need to start traning from this file. In most cases, there is no need to change it.

- **test_get_KP.py**: Code for getting the visualization and the values of disturbance.

- **MSTARdataset.py** and **read_dataset.py**: Code for reading data. 

- **data_transform.py**: Code for basic transformation used in the experiments.

- **unet.py** and **resnet.py**: Code for U-Net and ResNet structure.

- **KP_visual.py**: Code for get the visualization of KP. You can choose where the visual results saved in this file.

## Model Zoo

The trained models from the upstream tasks are available. The download address is as follows:

**We provide 3 models under each architecture, which are trained on TerraSAR-X (TSX) dataset, BigEarthNet (BEN) dataset and OpenSARUrban (OSU) dataset respectively.**

|Backbone | Input size | Pretrained model|Backbone | Input size | Pretrained model|
|-------- | ---------- | ----------|-------- | ---------- | ----------|
ResNet18 | 128×128 |  [baidu](https://pan.baidu.com/s/1nh-FTrVz7-LBev-fGpunPQ) (Extraction code:hy18)|MobileNetV3| 128×128 |  [baidu](https://pan.baidu.com/s/13Nvo8DCXszqlKgpzXWNR7A) (Extraction code:hymb)|
ResNet50 | 128×128 | [baidu](https://pan.baidu.com/s/1BXVR014Aecc9J4wZlOu1ew) (Extraction code:hy50)|DenseNet121 | 128×128 |  [baidu](https://pan.baidu.com/s/19pmJFoT35Wz2jemkuf6KPA) (Extraction code:hyde)|
ResNet101 | 128×128  | [baidu](https://pan.baidu.com/s/1OIQ5MFsmTWxiH-Smlb441g) (Extraction code:hy01)|Swin-T | 128×128 |  [baidu](https://pan.baidu.com/s/17hEe6251Yo63LKLI3PpTvg) (Extraction code:hyst)|
SENet50  | 128×128  | [baidu](https://pan.baidu.com/s/1rACPLIHdCxruFTVUhyipoQ) (Extraction code:hyse)|Swin-B | 128×128 |  [baidu](https://pan.baidu.com/s/1NlJfC4SnGFCotfwyl-za6Q) (Extraction code:hysb)|


## Contributors

In this repository, we implemented the ResNet series, DenseNet121, MobileNetV3, SENet50 and Swin series. The datasets we used contain TerraSAR-X, BigEarthNet-S1.0, openSARUrban, MSTAR, FuSARShip, OpenSARShip, SSDD, LS-SSDDv1.0, HRSID and SpaceNet6. Besides, we reimplemented FCOS on PyTorch based on MMDetection and Deeplabv3 based on MMSegmentation. Thanks for all the above works' contribution.

## Citation

If you find this repository useful for your publications, please consider citing our paper.