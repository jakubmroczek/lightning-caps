import torch
from torch import nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, epsilon: float = 0.0005):
        super(ReconstructionLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.epsilon = epsilon

    def forward(self, images, labels, classes, reconstructions):
        # labels and classes are not used, as they are not important in reconsturction process
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        result_loss = reconstruction_loss / images.size(0)
        return self.epsilon * result_loss