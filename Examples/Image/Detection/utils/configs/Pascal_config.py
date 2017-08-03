# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
__C.CNTK = edict()
cfg = __C

# data set config
__C.CNTK.DATASET = "Pascal"
__C.CNTK.MAP_FILE_PATH = "../../DataSets/Pascal/mappings"
__C.CNTK.CLASS_MAP_FILE = "class_map.txt"
__C.CNTK.TRAIN_MAP_FILE = "trainval2007.txt"
__C.CNTK.TRAIN_ROI_FILE = "trainval2007_rois_abs-xyxy_noPad_skipDif.txt"
__C.CNTK.TEST_MAP_FILE = "test2007.txt"
__C.CNTK.TEST_ROI_FILE = "test2007_rois_abs-xyxy_noPad_skipDif.txt"
__C.CNTK.NUM_TRAIN_IMAGES = 5010
__C.CNTK.NUM_TEST_IMAGES = 4952
__C.CNTK.PROPOSAL_LAYER_SCALES = [8, 16, 32]
