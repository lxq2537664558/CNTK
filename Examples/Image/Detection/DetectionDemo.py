import numpy as np
import demo_utils as od
from utils.config_helpers import merge_configs
import pdb

def get_configuration():
    from FasterRCNN.config import cfg as faster_rcnn_cfg
    cfg = merge_configs([faster_rcnn_cfg, {'DETECTOR': 'FasterRCNN'}])
    return cfg

if __name__ == '__main__':
    cfg = get_configuration()

    # train and test
    eval_model = od.train_object_detector(cfg)
    eval_results = od.evaluate_test_set(eval_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # detect objects in single image
    img_path = r"C:\src\CNTK\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_11_28_42_Pro.jpg"
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    # for i in fg_boxes[0]: print("box: {}, label: {}, score: {}".format(bboxes[i], labels[i], scores[i]))

    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)
