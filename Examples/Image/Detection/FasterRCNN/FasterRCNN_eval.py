import numpy as np
import cntk
from cntk import input_variable, Axis
from utils.nms_wrapper import apply_nms_to_single_image_results
from utils.cntk_helpers import regress_rois
import cv2 # pip install opencv-python


# Tests a Faster R-CNN model and plots images with detected boxes
def eval_and_plot_faster_rcnn(eval_model, num_images_to_plot, test_map_file, img_shape,
                              results_base_path, feature_node_name, classes,
                              drawUnregressedRois=False, drawNegativeRois=False,
                              nmsThreshold=0.5, nmsConfThreshold=0.0, bgrPlotThreshold = 0.8):
    # get image paths
    with open(test_map_file) as f:
        content = f.readlines()
    img_base_path = os.path.dirname(os.path.abspath(test_map_file))
    img_file_names = [os.path.join(img_base_path, x.split('\t')[1]) for x in content]

    # prepare model
    image_input = input_variable(img_shape, dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
    frcn_eval = eval_model(image_input, dims_input)

    #dims_input_const = cntk.constant([image_width, image_height, image_width, image_height, image_width, image_height], (1, 6))
    print("Plotting results from Faster R-CNN model for %s images." % num_images_to_plot)
    for i in range(0, num_images_to_plot):
        imgPath = img_file_names[i]

        # evaluate single image
        _, cntk_img_input, dims = load_resize_and_pad(imgPath, img_shape[2], img_shape[1])

        dims_input = np.array(dims, dtype=np.float32)
        dims_input.shape = (1,) + dims_input.shape
        output = frcn_eval.eval({frcn_eval.arguments[0]: [cntk_img_input], frcn_eval.arguments[1]: dims_input})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1).tolist()

        if drawUnregressedRois:
            # plot results without final regression
            imgDebug = visualizeResultsFaster(imgPath, labels, scores, out_rpn_rois, img_shape[2], img_shape[1],
                                              classes, nmsKeepIndices=None, boDrawNegativeRois=drawNegativeRois,
                                              decisionThreshold=bgrPlotThreshold)
            imsave("{}/{}_{}".format(results_base_path, i, os.path.basename(imgPath)), imgDebug)

        # apply regression and nms to bbox coordinates
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                    nms_threshold=nmsThreshold,
                                                    conf_threshold=nmsConfThreshold)

        img = visualizeResultsFaster(imgPath, labels, scores, regressed_rois, img_shape[2], img_shape[1],
                                     classes, nmsKeepIndices=nmsKeepIndices,
                                     boDrawNegativeRois=drawNegativeRois,
                                     decisionThreshold=bgrPlotThreshold)
        imsave("{}/{}_regr_{}".format(results_base_path, i, os.path.basename(imgPath)), img)

class FasterRCNN_Evaluator:
    def __init__(self, eval_model, cfg):
        # prepare model
        self._img_shape = (cfg["CNTK"].NUM_CHANNELS, cfg["CNTK"].IMAGE_HEIGHT, cfg["CNTK"].IMAGE_WIDTH)
        image_input = input_variable(shape=self._img_shape,
                                     dynamic_axes=[Axis.default_batch_axis()],
                                     name=cfg["CNTK"].FEATURE_NODE_NAME)
        dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
        self._eval_model = eval_model(image_input, dims_input)

    def process_image(self, img_path):
        _, cntk_img_input, dims = self._load_resize_and_pad(img_path, self._img_shape[2], self._img_shape[1])

        cntk_dims_input = np.array(dims, dtype=np.float32)
        cntk_dims_input.shape = (1,) + cntk_dims_input.shape
        output = self._eval_model.eval({self._eval_model.arguments[0]: [cntk_img_input],
                                        self._eval_model.arguments[1]: cntk_dims_input})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        return regressed_rois, out_cls_pred

    def _load_resize_and_pad(self, image_path, width, height, pad_value=114):
        if "@" in image_path:
            print("WARNING: zipped image archives are not supported for visualizing results.")
            exit(0)

        img = cv2.imread(image_path)
        img_width = len(img[0])
        img_height = len(img)
        scale_w = img_width > img_height
        target_w = width
        target_h = height

        if scale_w:
            target_h = int(np.round(img_height * float(width) / float(img_width)))
        else:
            target_w = int(np.round(img_width * float(height) / float(img_height)))

        resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)

        top = int(max(0, np.round((height - target_h) / 2)))
        left = int(max(0, np.round((width - target_w) / 2)))
        bottom = height - top - target_h
        right = width - left - target_w
        resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

        # transpose(2,0,1) converts the image to the HWC format which CNTK accepts
        model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

        dims = (width, height, target_w, target_h, img_width, img_height)
        return resized_with_pad, model_arg_rep, dims
