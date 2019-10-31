import torch.nn as nn
from datasets.data_channels import LidarChannels, ImageChannels

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha_classification, alpha_regression, alpha_depth, num_classes):
    	self._alpha_classification = alpha_classification
    	self._alpha_regression = alpha_regression
    	self._alpha_depth = alpha_depth
    	self._num_classes = num_classes

    	#Initialize losses
    	self._cross_entropy_loss = nn.CrossEntropyLoss()
    	self._l1_loss = nn.L1Loss()

    def forward(self, sample, header_out, img_out, img_chn):
    	assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"
    	class_scores, bboxes = header_out
    	class_scores = class_scores.squeeze().reshape((class_scores.shape[1], -1)).T.tolist() #Transpose and convert to list of (N_boxes * N_classes)
    	bboxes = bboxes.squeeze().reshape((bboxes.shape[1], -1)).T.tolist() #Transpose and convert to list of (N_boxes * N_features)

    	bboxes, class_scores = NMS(bboxes, class_scores)
    	transformed_bboxes = transformBBoxes(bboxes) 
    	gt_bboxes = sample.corresponding_groundtruth.getBoxes()
    	gt_sparse = sample.corresponding_groundtruth.getSparse(img_chn)

    	for gt_bbox in gt_bboxes:
    		closest_ind = self.getClosestBBox(transformed_bboxes, gt_bbox.center)
    		bbox = transformed_bboxes[closest_ind] #Corresponding output
    		class_score = class_scores[closest_ind] #Corresponding output

    		classification_loss = getClassificationLoss(torch.tensor(class_score), label)
    		regression_loss = getRegressionLoss(torch.tensor(bbox), self.getBBoxVector(gt_bbox))

    	depth_loss = getDepthLoss(img_out, gt_sparse)

    	#TODO 3D Bounding-box refinement loss
    	multi_task_loss = self._alpha_classification * classification_loss + self._alpha_regression * regression_loss + self._alpha_depth * depth_loss


    	return multi_task_loss

    def NMS(self, bboxes, class_scores, threshold=0.9):
    	assert threshold >= 0.0, "Non-maxima suppression must have a threshold of at least 0.0"
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

