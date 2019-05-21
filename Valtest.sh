#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=5 nohup python -u ./Source/main.py \
                       --gpu 1 --phase train \
                       --dataset FlyingThing \
                       --modelDir ./PAModel/ \
                       --auto_save_num 10 \
                       --imgNum 0 \
                       --valImgNum 215 \
                       --maxEpochs 1 \
                       --learningRate 0.001 \
                       --outputDir ./Result1/ \
                       --resultImgDir ./ResultImg_Val/ \
                       --trainListPath /home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2 \
                       --valListPath /home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2 \
                       --corpedImgWidth 1024 \
                       --corpedImgHeight 1024 \
                       --batchSize 1 \
                       --pretrain false > Valtest.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TrainDFC.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"