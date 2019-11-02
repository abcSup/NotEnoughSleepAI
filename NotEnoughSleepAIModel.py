import torch.nn as nn
from Backbone import Backbone

class NotEnoughSleepAIModel(nn.Module):
    def __init__(self, bev_height, img_chn, num_classes, anchor_dims):
        super(NotEnoughSleepAIModel, self).__init__()
        self.backbone = Backbone(bev_height, img_chn, num_classes)
        self.anchor_dims = anchor_dims

    def forward(self, sample, img_chn):
        #TODO 2D & 3D Refinement
        sparse_out, header_out = self.backbone(sample, img_chn)

        pred_objectness1, pred_objectness2, pred_class_scores1, pred_class_scores2, pred_bboxes1, pred_bboxes2, anchor_orients = header_out

        #Organize outputs
        #ANCHOR 1
        pred_bboxes_grid1 = torch.from_numpy((pred_bboxes1.shape[2:4]).reshape(2,-1).T) #Transpose and convert to (N_boxes * 2)
        pred_bboxes1 = pred_bboxes1.squeeze().reshape((pred_bboxes1.shape[1], -1)).T #Transpose and convert to (N_boxes * N_features)
        pred_class_scores1 = pred_class_scores1.squeeze().reshape((pred_class_scores1.shape[1], -1)).T #Transpose and convert to (N_boxes * N_classes)
        pred_objectness1 = pred_objectness1.squeeze().reshape((pred_objectness1.shape[1], -1)).T #Transpose and convert to (N_boxes * 1)

        #ANCHOR 2
        pred_bboxes_grid2 = torch.from_numpy((pred_bboxes2.shape[2:4]).reshape(2,-1).T) #Transpose and convert to (N_boxes * 2)
        pred_bboxes2 = pred_bboxes2.squeeze().reshape((pred_bboxes2.shape[1], -1)).T #Transpose and convert to (N_boxes * N_features)
        pred_class_scores2 = pred_class_scores2.squeeze().reshape((pred_class_scores2.shape[1], -1)).T #Transpose and convert to (N_boxes * N_classes)
        pred_objectness2 = pred_objectness2.squeeze().reshape((pred_objectness2.shape[1], -1)).T #Transpose and convert to (N_boxes * 1)

        #Suppress and transform, then project to BEV space
        #ANCHOR 1
        pred_bboxes1, pred_class_scores1, pred_bboxes_grid1, pred_objectness1 = NMS(pred_bboxes1, pred_class_scores1, pred_bboxes_grid1, pred_objectness1)
        pred_bboxes1 = transformBBoxes(pred_bboxes1, pred_bboxes_grid1, anchor_orients[0], project=True)

        #ANCHOR 2
        pred_bboxes2, pred_class_scores2, pred_bboxes_grid2, pred_objectness2 = NMS(pred_bboxes2, pred_class_scores2, pred_bboxes_grid2, pred_objectness2)
        pred_bboxes2 = transformBBoxes(pred_bboxes2, pred_bboxes_grid2, anchor_orients[1], project=True)

        anchor1 = (pred_bboxes1, pred_class_scores1)
        anchor2 = (pred_bboxes2, pred_class_scores2)

        return sparse_out, anchor1, anchor2

    def NMS(self, pred_bboxes, pred_class_scores, pred_bboxes_grid, pred_objectness, threshold=0.9):
        assert threshold >= 0.0, "Non-maxima suppression must have a threshold of >= 0.0"
        post_suppression_indices = np.where(pred_objectness >= threshold)

        return pred_bboxes[post_suppression_indices[0]], pred_class_scores[post_suppression_indices[0]], pred_bboxes_grid[post_suppression_indices[0]], pred_objectness[post_suppression_indices[0]]

    def transformBBoxes(self, pred_bboxes, pred_bboxes_grid, orient, project=True):
        #(t, x, y, z, l, w, h)
        transformed_bboxes = torch.zeros_like(pred_bboxes)

        #Add 0.5 as x, y pixel centre offset
        pred_bboxes_grid = pred_bboxes_grid + 0.5

        #Transformations: t_ as transformed bounding box features, a_ as anchor bounding box features, k_ as CNN output transformation constants
        #Bounding box centre transformation
        #t_x = sigma(k_x) + a_x 
        #t_y = sigma(k_y) + a_y 
        #t_z = sigma(k_x) (z_offset is 0 because all objects are assumed to be at street-level) 
        transformed_bboxes[:, 1:4] = torch.sigmoid(pred_bboxes[:, 1:4])
        transformed_bboxes[:, 1:3] += pred_bboxes_grid

        #Bounding box dimensions transformation
        #t_l = a_l * e^(k_l)
        #t_w = a_w * e^(k_w)
        #t_h = a_h * e^(k_h)
        transformed_bboxes[:, 4:] = self._anchor_dims * torch.exp(pred_bboxes[:, 4:])

        #Bounding box orientation transformation
        #t_t = a_t + sigma(k_t) * (pi / 2)
        transformed_bboxes[:, 0] = orient + torch.sigmoid(pred_bboxes[:, 0]) * (np.pi / 2)

        if project: #Project to BEV space
            transformed_bboxes = self.projectBBoxesToBev(transformed_bboxes)

        return transformed_bboxes

    def projectBBoxesToBev(self, pred_bboxes):
        #TODO project bounding box coordinates to BEV coordinates
        pass



