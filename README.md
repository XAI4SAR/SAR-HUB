# SAR-HUB

## 0. Table of Contents

* [Introduction](#1-introduction)
    * [Features](#11-features) 
    * [Contributions](#12-contributions)
* [Previously on SAR-HUB](#2-previously-on-sar-hub)
* [Getting Started](#3-getting-started)
    * [Requirements](#31-requirements)
    * [Pre-training](#32-pre-training)
        * [Data](#321-data-preparation)
        * [Initialization](#322-initialization)
        * [Optimization](#323-drae-and-mini-cbl)
    * [Fine-tuning](#33-fine-tuning)
        * [Model Hub](#331-model-hub)
        * [SAR Target Recognition](#332-sar-target-recognition)
        * [SAR Object Detection](#333-sar-object-detection)
        * [SAR Semantic Segmentation](#334-sar-semantic-segmentation)
    * [Explaining](#34-explaining)
* [Contributors](#4-contributors)
* [Citation](#5-citation)

## 1. Introduction

This project is for paper "SAR-HUB: Pre-training, Fine-tuning, and Explaining".

### 1.1 Features

1.  **Pre-training:** Deep neural networks are trained with large-scale, open-source SAR scene image datasets.
    
2.  **Fine-tuning:** The pre-trained DNNs are transferred to diverse SAR downstream tasks.
    
3.  **Explaining:** Benefits of SAR pre-trained models in comparison to optical pre-trained models are explained.
    
<img src="https://github.com/XAI4SAR/SAR-HUB/blob/main/img/intro.png" width="60%">

We release this repository with reproducibility (open-source code and datasets), generalization (sufficient experiments on different tasks), and explainability (qualitative and quantitative explanations).

### 1.2 Contributions

-   An optimization method for large-scale SAR image classification is proposed to improve model performance.
    
-   A novel explanation method is proposed to explain the benefits of SAR pre-trained models qualitatively and quantitatively.
    
-   The Model-Hub offers a variety of SAR pre-trained models validated on various SAR benchmark datasets.
    

## 2. Previously on SAR-HUB

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

-   To further improve the large-scale SAR scene classification performance and the feature generalization ability, we propose an optimization method with **dynamic range adapted Enhancement (DRAE)** and **mini-batch class imbalanced** **loss function** **(mini-CBL)**.
    
-   In pre-training, **seven** popular CNN and Transformer based architectures and **three** different large-scale SAR scene image datasets are explored, collected in Model-Hub. In fine-tuning, **seven** different SAR downstream tasks are evaluated.
    
-   We propose **SAR** **knowledge point (SAR-KP)**, together with CAM based methods, to explain why the SAR pre-trained models outperform ImageNet and optical remote sensing image pre-trained models in transfer learning.
    

## 3. Getting Started

### 3.1 Requirements

Please refer to [requirements](requirements) for installation.

If you need to conduct experiments of SAR scene classification, target recognition or SAR knowledge point, please download the required dependencies according to [here](XAI4SAR/SAR-HUB/requirements/scene_classification.txt).

If you need to conduct experiments of SAR object detection or sementic segmentation, please refer to [object_detection.txt](requirements/object_detection.txt) and [sementic_segmentation.txt](requirements/sementic_segmentation.txt) respectively. 

### 3.2 Pre-training

The file directory tree is as below:

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

#### 3.2.1 Data Preparation

  BigEarthNet-S1.0: https://bigearth.net/

  OpenSARUrban: https://pan.baidu.com/s/1D2TzmUWePYHWtNhuHL7KdQ

  Normalize the datasets to 0-1 and store them in dataset/xxx/xxx-npy folder with *npy* format. 

<!--   Of course, if you want to store data in your own style, then please change the *137th* and *93rd* lines of [datasets.py](SAR_scene_classification/src/dataset.py) according to the data path you store. -->

45872313
#### 3.2.2 Initialization



#### 3.2.3 DRAE and mini-CBL

<!--   We recommend you to use GPUs but not a CPU to train and test, because it will greatly shorten the time. -->

<!--   Before starting training, please change the parameters in *main.py*, including the dataset, model, path and so on. -->

  The usage of DRAE with Reinhard-Devlin:
  ```bash
  --xxx xx
  ```
  The usage of mini-CBL with Focal Loss: 
  ```bash
  --loss_type Mini_CB_FL
  ```
  
  An example of training OpenSARUrban with DRAE and mini-CBL using Multiple GPUs:
  
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main.py > result.txt
  ```
  
  If you want to use a single GPU, set *CUDA_VISIBLE_DEVICES* to the serial number of a single GPU and change *--nproc_per_node* to 1:
  
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 main.py > result.txt
  ```
<!--   The results will be written to *result.txt* when using *nohup*. If you want to observe the training process on the terminal, delete *nohup* and *> result.txt*. -->


<!-- 同train.py，把参数作为外部输入 -->

  ```bash
  python test.py --xxx xxx
  ```

<!--   **The explanation of significant code file or folder is as follows**:

  - **main.py**: Code for significant parameters. The main parameters are needed checking in this file. You need to start traning from this file.

  - **intial_multi_gpu.py**: Code for intializing the multi-gpu process. In most cases, there is no need to change it.

  - **dist_train.py** and **dist_val.py**: Code for training and validation in each epoch. In most cases, there is no need to change it.

  - **test.py**: Code for test for the final models.

  - **dataset.py** and **read_dataset.py**: Code for reading data of each datasets. You may need to change them because we use the *npy* format in experiments.

  - **loss.py**: Code for several losses used.

  - **transform.py** and **data_transform.py**: Code for several basic transformation used in the experiments.

  - **model_prepare.py**: Configuration code for loading models.

  - **models**: Base configuration folder for CNN and ViT structure code. In most cases, there is no need to change it. -->
  
### 3.3 Fine-tuning

#### 3.3.1 Model Hub

SAR pre-trained models are available as follows:

**We provide 3 models under each architecture, which are trained on TerraSAR-X (TSX) dataset, BigEarthNet (BEN) dataset and OpenSARUrban (OSU) dataset respectively.**

|Backbone | Input size | Pretrained model|Backbone | Input size | Pretrained model|
|-------- | ---------- | ----------|-------- | ---------- | ----------|
ResNet18 | 128×128 |  [baidu](https://pan.baidu.com/s/1nh-FTrVz7-LBev-fGpunPQ) (Extraction code:hy18)|MobileNetV3| 128×128 |  [baidu](https://pan.baidu.com/s/13Nvo8DCXszqlKgpzXWNR7A) (Extraction code:hymb)|
ResNet50 | 128×128 | [baidu](https://pan.baidu.com/s/1BXVR014Aecc9J4wZlOu1ew) (Extraction code:hy50)|DenseNet121 | 128×128 |  [baidu](https://pan.baidu.com/s/19pmJFoT35Wz2jemkuf6KPA) (Extraction code:hyde)|
ResNet101 | 128×128  | [baidu](https://pan.baidu.com/s/1OIQ5MFsmTWxiH-Smlb441g) (Extraction code:hy01)|Swin-T | 128×128 |  [baidu](https://pan.baidu.com/s/17hEe6251Yo63LKLI3PpTvg) (Extraction code:hyst)|
SENet50  | 128×128  | [baidu](https://pan.baidu.com/s/1rACPLIHdCxruFTVUhyipoQ) (Extraction code:hyse)|Swin-B | 128×128 |  [baidu](https://pan.baidu.com/s/1NlJfC4SnGFCotfwyl-za6Q) (Extraction code:hysb)|

#### 3.3.2 SAR Target Recognition

The file directory tree is as below:

  ```
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


**Data Preparation**

  FuSARShip: https://radars.ac.cn/web/data/getData?dataType=FUSAR

  MSTAR：https://www.sdms.afrl.af.mil/index.php?collection=mstar 

  OpenSARShip: https://opensar.sjtu.edu.cn/
  
  The train/test splitting settings used in our experiments can be found in data/FSS, data/MSTAR, and data/OSS

**Usage of SAR Pre-trained Models**

<!--   Before starting training, please change the parameters in *main.py*, including model type and so on.

  Then you can use the command below to start a training procedure: -->
  
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup python train.py > SENet_FuSARship_TSX.txt
  ```
  
<!--   这些细节没必要在readme里写。The results will be written to *SENet_FuSARship_TSX.txt* when using *nohup*. If you want to observe the training process on the terminal, delete *nohup* and *> SENet_FuSARship_TSX.txt*. -->
<!-- 
  **The explanation of significant code file or folder is as follows**:

  - **main.py**: Code for training and validation in each epoch. The main parameters are needed checking in this file. You need to start traning from this file.
 
  - **sampler.py**: Code for ImbalancedDatasetSampler.

  - **{}_dataset.py** and **read_dataset.py**: Code for reading data of each datasets. You may need to change them because we use the *npy* format in experiments.

  - **transform.py** and **data_transform.py**: Code for several basic transformation used in the experiments.


  - **models**: Base configuration folder for CNN and ViT structure code. In most cases, there is no need to change it. -->

#### 3.3.3 SAR Object Detection

The object detection are based on MMDetection framework,combining Feature Pyramid Networks (FPN) and Fully Convolutional One Stage (FCOS), and we have not changed any of it. Therefore, we only give the *SAR config* and *\_\_base\_\_* and introduce how to use them.

**Data Preparation**

  SSDD: https://drive.google.com/file/d/1grDw3zbGjQKYPjOxv9-h4WSUctoUvu1O/view

  HRSID: https://aistudio.baidu.com/aistudio/datasetdetail/54512
  
  LS-SSDDv1.0: https://radars.ac.cn/web/data/getData?newsColumnId=6b535674-3ce6-43cc-a725-9723d9c7492c

  The train/test splitting follow the official settings.

**Usage of SAR Pre-trained Models**

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

#### 3.3.4 SAR Semantic Segmentation

<!-- 同上修改 We adopt DeepLabv3 under MMSegmentation framework during the experiments. Similar to the object detection task, we give the *SAR config* and *\_\_base\_\_* and introduce how to use them.

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
  
  The results will be written to the log save path you set in each config file. -->

### 3.4 Explaining

<!--  按步骤来，每个步骤运行完，都能得到对应的结果，最好给出输出的例子（图例）-->

(1) U-Net explainer optimization:

```bash
xxx
```

(2) xxx

The code are proposed [here](SAR_KP).

<!-- #### **Train and Get_KP**

The ResNet-50 model used in KP is firstly trained on the MSTAR dataset. Then we connect it with U-Net to explain it. Before training, you need to change the parameters in the *train.py*, including the save path and the loaded model.

You can use the command below to start the training:

```bash
  python train.py
```
Notably, we don't use validation dataset during train, so you may need to use *tensorboard* or other training process visualization tool to check whether the training is normal by observing the loss curve. 

After training, you need to run the *test_Get_KP.py* to get KP. The visualization and the values of disturbance will be saved in *jpg* and *npy* format respectively.

If you want to get more intuitive visualization results, you can use *KP_visual.py* to colored the disturbance.
 -->
<!-- 
#### **Explanation of significant code file or folder**

- **train.py**: Code for training. The main parameters are needed checking in this file. You need to start traning from this file. In most cases, there is no need to change it.

- **test_get_KP.py**: Code for getting the visualization and the values of disturbance.

- **MSTARdataset.py** and **read_dataset.py**: Code for reading data. 

- **data_transform.py**: Code for basic transformation used in the experiments.

- **unet.py** and **resnet.py**: Code for U-Net and ResNet structure.

- **KP_visual.py**: Code for get the visualization of KP. You can choose where the visual results saved in this file. -->


## 4. Contributors

In this repository, we implemented the ResNet series, DenseNet121, MobileNetV3, SENet50 and Swin series. The datasets we used contain TerraSAR-X, BigEarthNet-S1.0, openSARUrban, MSTAR, FuSARShip, OpenSARShip, SSDD, LS-SSDDv1.0, HRSID and SpaceNet6. Besides, we reimplemented FCOS on PyTorch based on MMDetection and Deeplabv3 based on MMSegmentation. Thanks for all the above works' contribution.

## 5. Citation

If you find this repository useful for your publications, please consider citing our paper.
