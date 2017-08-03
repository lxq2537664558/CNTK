import os
import numpy as np
from FasterRCNN_train import prepare, train_faster_rcnn
from FasterRCNN_eval import compute_test_set_aps, plot_test_set_results
from utils.config_helpers import merge_configs

# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    # set parameters
    from config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg
    cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg])
    prepare(cfg)

    # train and test
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # Plot results on test set images
    if cfg["CNTK"].VISUALIZE_RESULTS:
        num_eval = min(cfg["CNTK"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg["CNTK"].OUTPUT_PATH, cfg["CNTK"].DATASET)
        plot_test_set_results(trained_model, num_eval, results_folder, cfg)
