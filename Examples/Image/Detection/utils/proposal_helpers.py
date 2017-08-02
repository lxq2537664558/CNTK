import numpy as np
from dlib import find_candidate_object_locations

random_seed = 23

def compute_proposals(img, num_proposals, min_w, min_h):
    all_rects = []
    min_size = min_w * min_h
    find_candidate_object_locations(img, all_rects, min_size=min_size)

    rects = []
    for k, d in enumerate(all_rects):
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        if w < min_w or h < min_h:
            continue
        rects.append([d.left(), d.top(), d.right(), d.bottom()])

    np_rects = np.array(rects)
    num_rects = np_rects.shape[0]
    np.random.seed(random_seed)
    if num_rects < num_proposals:
        img_w = len(img[0])
        img_h = len(img)
        grid_proposals = compute_grid_proposals(num_proposals - len(rects), img_w, img_h, min_w, min_h)
        np_rects = np.vstack([np_rects, grid_proposals])
    elif len(rects) > num_proposals:
        keep_inds = range(num_rects)
        keep_inds = np.random.choice(keep_inds, size=num_proposals, replace=False)
        np_rects = np_rects[keep_inds]

    return np_rects

def compute_grid_proposals(num_proposals, img_w, img_h, min_w, min_h, max_w=None, max_h=None, aspect_ratios = [1.0], shuffle=True):
    min_wh = max(min_w, min_h)
    max_wh = min(img_h, img_w) / 2
    if max_w is not None: max_wh = min(max_wh, max_w)
    if max_h is not None: max_wh = min(max_wh, max_h)

    rects = []
    iter = 0
    while len(rects) < num_proposals:
        new_ar = []
        for ar in aspect_ratios:
            new_ar.append(ar * (0.9 ** iter))
            new_ar.append(ar * (1.1 ** iter))

        new_rects = _compute_grid_proposals(img_w, img_h, min_wh, max_wh, new_ar)
        take = min(num_proposals - len(rects), len(new_rects))
        new_rects = new_rects[:take]
        rects.extend(new_rects)

    np_rects = np.array(rects)
    num_rects = np_rects.shape[0]
    if shuffle and num_proposals < num_rects:
        keep_inds = range(num_rects)
        keep_inds = np.random.choice(keep_inds, size=num_proposals, replace=False)
        np_rects = np_rects[keep_inds]
    else:
        np_rects = np_rects[:num_proposals]

    return np_rects

def _compute_grid_proposals(img_w, img_h, min_wh, max_wh, aspect_ratios):
    rects = []
    cell_w = max_wh
    while cell_w >= min_wh:
        step = cell_w / 2.0
        for aspect_ratio in aspect_ratios:
            w_start = 0
            while w_start < img_w:
                h_start = 0
                while h_start < img_h:
                    if aspect_ratio < 1:
                        w_end = w_start + cell_w
                        h_end = h_start + cell_w / aspect_ratio
                    else:
                        w_end = w_start + cell_w * aspect_ratio
                        h_end = h_start + cell_w
                    if w_end < img_w-1 and h_end < img_h-1:
                        rects.append([int(w_start), int(h_start), int(w_end), int(h_end)])
                    h_start += step
                w_start += step
        cell_w = cell_w / 2

    return rects

class ProposalProvider:
    def __init__(self, proposal_list, proposal_cfg=None):
        self.proposal_list = proposal_list
        self.proposal_cfg = proposal_cfg

    @classmethod
    def fromfile(cls, filename):
        proposal_list = [] # TODO: read proposals from file
        return cls(proposal_list)

    @classmethod
    def fromconfig(cls, proposal_cfg):
        if proposal_cfg.BUFFER_IN_MEMORY:
            proposal_list = [] # TODO: compute_proposals
            return cls(proposal_list)
        else:
           return cls(None, proposal_cfg)

    def load_from_file(self, filename):
        return None

    def write_to_file(self, filename):
        return None

    def get_scaled_proposals(self, index, img_stats):
        return self.proposal_list[index]

    def compute_scaled_proposals(self, img, img_stats):
        return None # TODO: compute based on self.proposal_cfg

if __name__ == '__main__':
    # TODO: reading the image will be done externally
    import cv2

    image_file = r"C:\src\CNTK\Examples\Image\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000015.jpg"
    img = cv2.imread(image_file)

    # 0.18 sec for 4000
    # 0.15 sec for 2000
    # 0.13 sec for 1000
    num_proposals = 2000
    num_runs = 100
    import time
    start = int(time.time())
    for i in range(num_runs):
        proposals = compute_proposals(img, num_proposals, 20, 20)
    total = int(time.time() - start)
    print ("time: {}".format(total / (1.0 * num_runs)))

    assert len(proposals) == num_proposals, "{} != {}".format(len(proposals), num_proposals)
