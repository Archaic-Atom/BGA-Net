#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=5 nohup python -u ./Source/main.py \
                       --gpu 1 --phase test \
                       --dataset FlyingThing \
                       --modelDir ./PAModel/ \
                       --auto_save_num 1 \
                       --imgNum 50 \
                       --valImgNum 0 \
                       --maxEpochs 1000 \
                       --learningRate 0.001 \
                       --outputDir ./Result1/ \
                       --trainListPath /home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2 \
                       --testListPath /home2/Documents/DFC2019_track2_trainval/Test-Track2/ \
                       --corpedImgWidth 1024 \
                       --corpedImgHeight 1024 \
                       --padedImgWidth 1024 \
                       --padedImgHeight 1024 \
                       --batchSize 1 \
                       --pretrain false > TestDFC.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TrainDFC.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"