# -*- coding: utf-8 -*-
from Define import *
from PIL import Image
import random
import cv2


def Vertical_Flip(imgL, imgR, disp_gt, cls_gt):
    flip_prop = np.random.randint(low=0, high=2)

    if flip_prop == 0:
        imgL = cv2.flip(imgL, 0)
        imgR = cv2.flip(imgR, 0)
        disp_gt = cv2.flip(disp_gt, 0)
        cls_gt = cv2.flip(cls_gt, 0)

    return imgL, imgR, disp_gt, cls_gt


def Horizontal_Flip(img, imgGround, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=3)
    if flip_prop == 0:
        img = cv2.flip(img, 0)
        imgGround = cv2.flip(imgGround, 0)
    elif flip_prop == 1:
        img = cv2.flip(img, 1)
        imgGround = cv2.flip(imgGround, 1)

    flip_prop = np.random.randint(low=0, high=3)
    if flip_prop == 0:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        imgGround = cv2.transpose(imgGround)
        imgGround = cv2.flip(imgGround, 0)
    elif flip_prop == 1:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        imgGround = cv2.transpose(imgGround)
        imgGround = cv2.flip(imgGround, 1)

    return img, imgGround


def NormalizeRGB(img):
    img = img.astype(float)
    for i in range(IMG_DEPTH):
        minval = img[:, :, i].min()
        maxval = img[:, :, i].max()
        if minval != maxval:
            img[:, :, i] = (img[:, :, i]-minval)/(maxval-minval)
    return img


def ImgProcessing(img):
    img = NormalizeRGB(img)
    img = img[:, :, :]
    return img


# Slice
def ImgSlice(img, x, y, w, h):
    return img[y:y+h, x:x+w, :]


def ImgGroundSlice(img, x, y, w, h):
    return img[y:y+h, x:x+w]


def RandomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image
