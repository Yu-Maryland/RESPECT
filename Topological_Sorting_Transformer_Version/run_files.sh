#!/bin/bash

models='densenet121 densenet169 densenet201 inception_resnet_v2 inception_v3 mobilenet_1.00_224 mobilenetv2_1.00_224 resnet101 resnet101v2 resnet152 resnet152v2 resnet50 resnet50v2 vgg16 vgg19 xception NASNet'
#model="NASNet"

#distributor="run_20210821T160800/epoch-296.pt" # inDim15 baseline-157
distributor="run_20210831T103517/epoch-296.pt" # inDim15 baseline-157

#CUDA_VISIBLE_DEVICES=2 python run.py --eval_only --load_path outputs/toposort_30/${distributor} --eval_dataset_path ../dataset/eval_dataset/model_layer_embedding_indegree_6/${model}.pt --graph_file ${model}.txt

for model in $models
do
    #CUDA_VISIBLE_DEVICES=0 python run.py --eval_only --load_path outputs/toposort_30/${distributor} --eval_dataset_path ../dataset/eval_dataset/model_layer_embedding_indegree_6/${model}.pt --graph_file ${model}.txt
    CUDA_VISIBLE_DEVICES=1 python run.py --eval_only --load_path outputs/toposort_30/${distributor} --eval_dataset_path ../dataset/eval_dataset/model_layer_embedding_indegree_6/${model}.pt --graph_file ${model}.txt
done



