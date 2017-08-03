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
__C.CNTK.BASE_MODEL = "AlexNet"
__C.CNTK.BASE_MODEL_FILE = "AlexNet.model"
__C.CNTK.IMG_PAD_COLOR = [114, 114, 114]
__C.CNTK.FEATURE_NODE_NAME = "features"
__C.CNTK.LAST_CONV_NODE_NAME = "conv5.y"
__C.CNTK.START_TRAIN_CONV_NODE_NAME = __C.CNTK.FEATURE_NODE_NAME
__C.CNTK.POOL_NODE_NAME = "pool3"
__C.CNTK.LAST_HIDDEN_NODE_NAME = "h2_d"
__C.CNTK.RPN_NUM_CHANNELS = 256
__C.CNTK.ROI_DIM = 6
__C.CNTK.E2E_LR_FACTOR = 1.0
__C.CNTK.RPN_LR_FACTOR = 1.0
__C.CNTK.FRCN_LR_FACTOR = 1.0
