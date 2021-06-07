#!/bin/bash

for arch in resnet56 vgg16bn wideresnet32_4 mobilenet; do
    for dataset in cifar10 cifar100 tinyimagenet; do
        sh ./single_arch.sh $arch $dataset
    done
done

sh ./single_arch.sh tv_resnet imagenet