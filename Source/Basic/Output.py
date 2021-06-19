# -*- coding: utf-8 -*-
from Define import *
from PIL import Image
import cv2
import tifffile
from copy import deepcopy


# new folder
def Mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    # check the file pat
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    return


# Create the result file
def CreateResultFile(args):
    # create the dir
    Mkdir(args.outputDir)
    Mkdir(args.resultImgDir)

    if args.phase == 'train':
        # four file
        train_loss = args.outputDir + TRAIN_LOSS_FILE
        train_acc = args.outputDir + TRAIN_ACC_FILE
        val_acc = args.outputDir + VAL_ACC_FILE
        val_loss = args.outputDir + VAL_LOSS_FILE

        # if it is a new train model
        if args.pretrain:
            if os.path.exists(val_acc):
                os.remove(val_acc)
            if os.path.exists(train_loss):
                os.remove(train_loss)
            if os.path.exists(train_acc):
                os.remove(train_acc)
            if os.path.exists(val_loss):
                os.remove(val_loss)

        fd_train_acc = open(train_acc, 'a')
        fd_train_loss = open(train_loss, 'a')
        fd_val_acc = open(val_acc, 'a')
        fd_val_loss = open(val_loss, 'a')

        return fd_train_acc, fd_train_loss, fd_val_acc, fd_val_loss
    else:
        test_acc = args.outputDir + TEST_ACC_FILE
        if os.path.exists(test_acc):
            os.remove(test_acc)
        return open(test_acc, 'w')


# create the output file
def GenerateOutImgPath(dirPath, filenameFormat, imgType, num):
    return dirPath + filenameFormat % num + imgType


# save the data
def SaveTestData(args, resImg, num):
    path = GenerateOutImgPath(args.resultImgDir, args.saveFormat, args.imgType, num)
    imgArray = DepthToImgArray(resImg)
    SavePngImg(imgArray, path)


# write file
def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


# change the data
def DepthToImgArray(mat):
    mat = np.array(mat)
    return (mat * float(DEPTH_DIVIDING)).astype(np.uint16)


# save the png file
def SavePngImg(img, path):
    cv2.imwrite(path, img)


# save image
def SaveImg(img, path):
    img.save(path)


# save from the img
def SaveArray2Img(mat, path):
    img = Image.fromarray(mat)
    SaveImg(img, path)


def SaveDFCTestImg(args, resImg, name):
    resImg = np.array(resImg)

    dsp_name = args.resultImgDir + name + 'LEFT_DSP.tif'
    viz_name = args.resultImgDir + name + 'STEREO_GRAY.tif'

    tifffile.imsave(dsp_name, resImg, compress=6)

    # save grayscale version of image for visual inspection
    resImg = resImg - resImg.min()
    resImg = ((resImg / resImg.max()) * 255.0).astype(np.uint8)
    resImg = cv2.cvtColor(resImg, cv2.COLOR_GRAY2RGB)
    tifffile.imsave(viz_name,  resImg, compress=6)
