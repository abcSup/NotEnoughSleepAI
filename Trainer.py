import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from os.path import join
from datasets.data_channels import LidarChannels, ImageChannels
from NotEnoughSleepAIModel import NotEnoughSleepAIModel
from MultiTaskLoss import MultiTaskLoss

lr = 1e-4
momentum = 0.9

class Trainer(nn.Module):
    def __init__(self, device, bev_height=30, img_in_chn=4, num_classes=23, alpha_classification=1.0, alpha_regression=1.0, alpha_depth=1.0):
        super(Trainer, self).__init__()

        self.model = NotEnoughSleepAIModel(bev_height, img_in_chn, num_classes).to(device)
        self.optim = optim.SGD([
                        {'params': self.model.parameters()},
                     ], lr=lr, momentum=momentum)

        self.multi_task_loss = MultiTaskLoss(alpha_classification, alpha_regression, alpha_depth)
        self.metrics = {}

    def update(self, sample):
        loss = 0.0
        for chn in ImageChannels:
            channel = chn.value

            sparse_out, transformed_bboxes, pred_class_scores = self.model(sample, channel)
            loss += self.multi_task_loss(sample, sparse_out, transformed_bboxes, pred_class_scores, channel)

        self.model.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {
            "loss/total_loss": loss.item(),
        }
        self.metrics = metrics

        return sparse_out, transformed_bboxes, pred_class_scores

    def get_metrics(self):
        return self.metrics

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['weight'])

    def save(self, save_dir, iterations):
        weight_fn = join(save_dir, "not_enough_sleep_%d.pkl" % iterations)

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': iterations,
        }

        torch.save(state, weight_fn)

