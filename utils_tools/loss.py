from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(AttackLoss, self).__init__()
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    # 实际检测中输出的bound boxes, 是已经解码后的结果， 来自detection
    #det_boxes, det_labels, det_scores

    def forward(self, det_boxes, det_scores, det_labels, boxes, labels):
        """
        Forward propagation.

        :param det_boxes: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (n_objects,4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        #将预测框等同于先验框
        assert len(det_boxes) == len(det_scores) and len(boxes) == len(labels)
        n_classes = len(label_map)
        
        #true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        #true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)
        
        # For each image
        for i in range(1):
            n_objects = boxes[i].shape[0]
            n_detection = det_boxes[i].shape[0]

            overlap = find_jaccard_overlap(boxes[i],
                                           det_boxes[i])  # (n_objects, n_detection)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_detection, object_for_each_detection = overlap.max(dim=0)  # (n_detection), unit: object id, each detection match a object with maximum overlap

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the detection that has the maximum overlap for each object.
            _, detection_for_each_object = overlap.max(dim=1)  # (N_o) each object match a detection with maximum overlap

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_detection[detection_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_detection[detection_for_each_object] = 1.

            # Labels for each prior
            label_for_each_detection = labels[i][object_for_each_detection]  # (n_detection)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_detection[overlap_for_each_detection < self.threshold] = 0  # (8732)

            # Store
            # true_classes.append(label_for_each_detection)
            true_class = label_for_each_detection
            # Encode center-size object coordinates into the form we regressed predicted boxes to
            # true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_detection]), self.priors_cxcy)  # (8732, 4)
            true_loc = boxes[i][object_for_each_detection]
            # true_locs.appen(loc)

            # Identify priors that are positive (object/non-background)
            positive_priors = true_class != 0  # (n_detection)
            
            # LOCALIZATION LOSS
            # Localization loss is computed only over positive (non-background) priors
            loc_loss = self.smooth_l1(det_boxes[i][positive_priors], true_loc[positive_priors])  # (), scalar

            # CONFIDENCE LOSS

            # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
            # That is, FOR EACH IMAGE,
            # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
            # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

            # Number of positive and hard-negative priors per image， 正负样本比例
            n_positives = positive_priors.sum(dim=0)  # (N)
            n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        
            # First, find the loss for all priors
            #conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
            # conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)
            
            conf_loss_all = 1 - det_scores[i]
            
            # We already know which priors are positive
            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

            # Next, find which priors are hard-negative
            # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg, _ = conf_loss_neg.sort(dim=0, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_detection)).to(device)
            hard_negatives = hardness_ranks < n_hard_negatives
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
