import numpy as np
import cntk
from cntk import input_variable, Axis
from utils.nms_wrapper import apply_nms_to_single_image_results
from utils.plot_helpers import load_resize_and_pad
from utils.cntk_helpers import regress_rois


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
        _, cntk_img_input, dims = load_resize_and_pad(img_path, self._img_shape[2], self._img_shape[1])

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
