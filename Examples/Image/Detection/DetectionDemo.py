import os
import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN.config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': 'FasterRCNN'}])

if __name__ == '__main__':
    cfg = get_configuration()

    # train and test
    eval_model = od.train_object_detector(cfg)
    eval_results = od.evaluate_test_set(eval_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # detect objects in single image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\DataSets\Grocery\testImages\WIN_20160803_11_28_42_Pro.jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    # for i in fg_boxes[0]: print("box: {}, label: {}, score: {}".format(bboxes[i], labels[i], scores[i]))

    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)
