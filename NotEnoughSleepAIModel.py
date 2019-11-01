import torch.nn as nn
from Backbone import Backbone

class NotEnoughSleepAIModel(nn.Module):
    def __init__(self, bev_height, img_chn, num_classes):
        super(NotEnoughSleepAIModel, self).__init__()
        self.backbone = Backbone(bev_height, img_chn, num_classes)

    def forward(self, sample, img_chn):
        #TODO 2D & 3D Refinement
        sparse_out, header_out = self.backbone(sample, img_chn)

        pred_class_scores, pred_bboxes = header_out
        pred_class_scores = pred_class_scores.squeeze().reshape((pred_class_scores.shape[1], -1)).T.tolist() #Transpose and convert to list of (N_boxes * N_classes)
        pred_bboxes = pred_bboxes.squeeze().reshape((pred_bboxes.shape[1], -1)).T.tolist() #Transpose and convert to list of (N_boxes * N_features)

        pred_bboxes, pred_class_scores = NMS(pred_bboxes, pred_class_scores)
        transformed_bboxes = transformBBoxes(bboxes) 

        return sparse_out, transformed_bboxes, pred_class_scores

    def NMS(self, bboxes, class_scores, threshold=0.9):
        assert threshold >= 0.0, "Non-maxima suppression must have a threshold of >= 0.0"
        #Class scores in list of 100 * (object/no_object + num_classes)
        #Bboxes in list of 100 * (num_features)
        post_suppression_indices = np.where(np.array(class_scores)[:, 0] >= threshold)

        return bboxes[post_suppression_indices[0]], class_scores[post_suppression_indices[0]]

    def transformBBox(self, bbox):
        #TODO implement bounding-box transformation equations
        #return transformed_bbox
        pass

    def transformBBoxes(self, bboxes):
        transformed_bboxes = []
        for bbox in bboxes:
            transformed_bboxes += self.transformBBox(bbox)

        return transformed_bboxes