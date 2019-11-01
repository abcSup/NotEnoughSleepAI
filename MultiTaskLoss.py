import torch.nn as nn
from datasets.data_channels import LidarChannels, ImageChannels

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes, alpha_classification=1.0, alpha_regression=1.0, alpha_depth=1.0):
        self._num_classes = num_classes
        self._alpha_classification = alpha_classification
        self._alpha_regression = alpha_regression
        self._alpha_depth = alpha_depth

        #Initialize losses
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._l1_loss = nn.L1Loss()

    def forward(self, sample, pred_sparse, pred_bboxes, pred_class_scores, img_chn):
        assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"
        gt_bboxes = sample.corresponding_groundtruth.getBoxes()
        gt_sparse = sample.corresponding_groundtruth.getSparse(img_chn)

        for gt_bbox in gt_bboxes:
            closest_ind = self.getClosestBBox(pred_bboxes, gt_bbox.center)
            pred_bbox = pred_bboxes[closest_ind] #Corresponding output
            pred_class_score = pred_class_scores[closest_ind] #Corresponding output
            gt_label = gt_bbox.label

            classification_loss = getClassificationLoss(torch.tensor(pred_class_score), gt_label)
            regression_loss = getRegressionLoss(torch.tensor(pred_bbox), self.getBBoxVector(gt_bbox))

        depth_loss = getDepthLoss(pred_sparse, gt_sparse)

        #TODO 3D Bounding-box refinement loss
        multi_task_loss = self._alpha_classification * classification_loss + self._alpha_regression * regression_loss + self._alpha_depth * depth_loss

        return multi_task_loss

    def getClosestBBox(self, bboxes, center_xyz):
        #bbox as (dx, dy, dz, w, l, h, t)

        euclidean_distance = np.array(a)[:, 0:3] - center_xyz
        return np.argmin(euclidean_distance)

    def getBBoxVector(self, gt_bbox):
        center = gt_bbox.center #Center in X,Y,Z coordinates
        wlh = gt_bbox.wlh #Width, length, height of bbox
        orientation = np.array(gt_bbox.orientation) #Yaw of bbox

        return torch.tensor(np.concatenate(center, wlh, orientation))


    def getClassificationLoss(predicted_class_scores, ground_truth_label):
        target = torch.zeros(self._num_classes)
        target[ground_truth_label] = 1

        return self._cross_entropy_loss(np.array(predicted_class_scores)[1:], target)

    def getRegressionLoss(predicted_bbox, ground_truth_bbox):
        return self._l1_loss(predicted_bbox, ground_truth_bbox)

    def getDepthLoss(predicted_depth, ground_truth_depth):
        return self._l1_loss(predicted_depth, ground_truth_depth)

