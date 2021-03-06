# -*- coding: utf-8 -*-
from Basic.Define import *
from BasicModule import *
from Basic.LogHandler import *
import Basic.Config as conf
import math


class MSGFNet(object):
    def __init__(self):
        pass

    def NetWork(self, imgL, imgR, height, width, training=True):
        with tf.variable_scope("MSGFNet"):
            Info('├── Begin Build ExtractUnaryFeature')
            with tf.variable_scope("ExtractUnaryFeature") as scope:
                imgL_feature = ExtractUnaryFeatureModule(imgL, training=training)
                scope.reuse_variables()
                imgR_feature = ExtractUnaryFeatureModule(imgR, training=training)
            Info('│   └── After ExtractUnaryFeature:' + str(imgL_feature.get_shape()))

            Info('├── Begin Build Cost Volume')
            cost_vol = BuildCostVolumeModule(imgL_feature, imgR_feature,
                                             IMG_DISPARITY, training=training)
            Info('│   └── After Cost Volume:' + str(cost_vol.get_shape()))

            Info('├── Begin Build 3DMatching')
            coarse_map = MatchingModule(cost_vol, training=training)
            Info('│   └── After 3DMatching:' + str(coarse_map.get_shape()))

            Info('├── Begin Build CLS Recovery')
            coarse_cls_map = SegModule(imgL_feature, LABLE_NUM, training=training)
            Info('│   └── After CLSRecovery:' + str(coarse_cls_map.get_shape()))

            Info('├── Begin Build DispMapRefine')
            refine_map = DispRefinementModule(coarse_map, imgL,
                                              coarse_cls_map, training=training)
            Info('│   └── After DispMapRefine:' + str(refine_map.get_shape()))

            Info('└── Begin Build CLSMapRefine')
            refine_cls_map = ClsRefinementModule(
                coarse_cls_map, imgL, coarse_map, training=training)
            Info('    └── After CLSMapRefine:' + str(refine_cls_map.get_shape()))

        return coarse_cls_map, refine_cls_map, coarse_map, refine_map
