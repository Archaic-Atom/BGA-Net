# -*- coding: utf-8 -*-
from JackBasicStructLib.JackBasicLayer import *
from JackBasicStructLib.JackClassicalBlock import *
from BasicBlock import *


def ExtractUnaryFeatureModule(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureModule"):
        x = ExtractUnaryFeatureBlock1(x, training=training)
        output_raw = ExtractUnaryFeatureBlock2(x, training=training)
        output_skip_1 = ExtractUnaryFeatureBlock3(output_raw, training=training)
        output_skip_2 = ExtractUnaryFeatureBlock4(output_skip_1, training=training)
        x = ASPPBlock(output_skip_2, 32, "ASPP", training=training)
        x = tf.concat([output_raw, output_skip_1, output_skip_2, x], axis=3)
        x = ExtractUnaryFeatureBlock5(x, training=training)
    return x


def BuildCostVolumeModule(imgL, imgR, disp_num, training=True):
    with tf.variable_scope("BuildCostVolume") as scope:
        imgL = ExtractCostFeatureBlock(imgL, training=training)
        scope.reuse_variables()
        imgR = ExtractCostFeatureBlock(imgR, training=training)

        cost_vol = BuildCostVolumeBlock(imgL, imgR, disp_num)
    return cost_vol


def MatchingModule(x, training=True):
    with tf.variable_scope("MatchingModule"):
        shortcut = x
        x, level_list = EncoderBlock(x, training=training)
        x = NonLocalGroupBlock(x, training=training)
        x = DecoderBlock(x, level_list, training=training)
        x = shortcut + x
        x = RecoverSizeBlock(x, training=training)
        x = SoftArgMinBlock(x)
    return x


def DispRefinementModule(x, imgL, seg, training=True):
    with tf.variable_scope("RefinementModule"):
        shortcut = x
        x = FeatureConcat(x, imgL, seg, training=training)
        x = ResidualLearning(x, training=training)
        x = shortcut + x
    return x


def SegModule(x, cls_num, training=True):
    with tf.variable_scope("SegModule"):
        shortcut = ExtractSegFeatureBlock(x, training=training)
        x = SPPBlock(shortcut, 32, "SPP", training=training)
        x = tf.concat([x, shortcut], axis=3)
        x = RecoveryCLSSizeBlock(x, cls_num, training=training)
    return x


def ClsRefinementModule(x, imgL, disp, training=True):
    with tf.variable_scope("ClsRefinementModule"):
        shortcut = x
        cls_num = x.get_shape()[3]
        x = ClsFeatureConcat(x, imgL, disp, training=training)
        x = ClsResidualLearning(x, cls_num, training=training)
        x = shortcut + x

    return x
