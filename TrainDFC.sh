#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=5,6 nohup python -u ./Source/main.py \
                      --gpu 2 --phase train \
                      --dataset FlyingThing \
                      --modelDir ./PAModel/ \
                      --auto_save_num 1 \
                      --imgNum 4077 \
                      --valImgNum 0 \
                      --maxEpochs 300 \
                      --learningRate 0.001 \
                      --outputDir ./Result/ \
                      --trainListPath /home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2 \
                      --corpedImgWidth 384 \
                      --corpedImgHeight 384 \
                      --batchSize 2 \
                      --pretrain false > TrainDFC.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TrainDFC.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
