#!/bin/bash

models='resnet101 densenet121 densenet169 densenet201 mobilenet_1.00_224 mobilenetv2_1.00_224 resnet101 resnet101v2 resnet152 resnet152v2'
distributor="saved_model/run_20210827T155340/epoch-298.pt" 

for model in $models
do
    CUDA_VISIBLE_DEVICES=0 python run.py --eval_only --load_path ${distributor} --eval_dataset_path RESPECT_Eval_ImageNet_Models/${model}.pt --graph_file ${model}.txt
done



