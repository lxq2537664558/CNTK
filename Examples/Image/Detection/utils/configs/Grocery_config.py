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
__C.CNTK.DATASET = "Grocery"
__C.CNTK.MAP_FILE_PATH = "../../DataSets/Grocery"
__C.CNTK.CLASS_MAP_FILE = "class_map.txt"
__C.CNTK.TRAIN_MAP_FILE = "train_img_file.txt"
__C.CNTK.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.CNTK.TEST_MAP_FILE = "test_img_file.txt"
__C.CNTK.TEST_ROI_FILE = "test_roi_file.txt"
__C.CNTK.NUM_TRAIN_IMAGES = 20
__C.CNTK.NUM_TEST_IMAGES = 5
__C.CNTK.PROPOSAL_LAYER_SCALES = [4, 8, 12]
