#!/bin/bash
if [ "$1" == "train" ]; then
    echo "Training mode"
    python train_ssd.py  --regression_loss rql --dataset_name voc07  --model_type shadow
elif [ "$1" == "attack" ]; then
    echo "Attack mode"
    python mia_ssd.py --split non_member --regression_loss rel --dataset_name voc07+12
    # 在这里添加你的测试代码
else
    echo "Invalid argument. Please use 'train' or 'attack'"
fi