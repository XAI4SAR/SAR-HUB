#!/bin/bash

task=$1
if [ $task = "train" ];then
CUDA_VISIBLE_DEVICES=1 nohup python train.py > /DATA/yhd/pycharm_YHD/transfer/transfer/target_recognition/train/SENet_FuSARship_IMG.txt
fi
done 
