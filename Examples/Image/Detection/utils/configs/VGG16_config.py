# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
__C.CNTK = edict()
cfg = __C

# model config
__C.CNTK.BASE_MODEL = "VGG16"
__C.CNTK.BASE_MODEL_FILE = "VGG16_ImageNet_Caffe.model"
__C.CNTK.IMG_PAD_COLOR = [103, 116, 123]
__C.CNTK.FEATURE_NODE_NAME = "data"
__C.CNTK.LAST_CONV_NODE_NAME = "relu5_3"
__C.CNTK.START_TRAIN_CONV_NODE_NAME = "pool2" # __C.CNTK.FEATURE_NODE_NAME
__C.CNTK.POOL_NODE_NAME = "pool5"
__C.CNTK.LAST_HIDDEN_NODE_NAME = "drop7"
__C.CNTK.FEATURE_STRIDE = 16
__C.CNTK.RPN_NUM_CHANNELS = 512
__C.CNTK.ROI_DIM = 7
__C.CNTK.E2E_LR_FACTOR = 1.0
__C.CNTK.RPN_LR_FACTOR = 1.0
__C.CNTK.FRCN_LR_FACTOR = 1.0

