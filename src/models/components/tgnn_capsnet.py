
from math import floor
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

NUM_ROUTING_ITERATIONS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class GraphLayer(nn.Module):
    def forward(self, x):
        X = []
        A = []
        return X,A

class TgnnLayer():
    pass

class TgnnCapsuleLayer():
    pass

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).to(DEVICE)
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class TgnnCapNet(nn.Module):
    def __init__(
        self, 
        first_capsule_layer_dimension:int = 8,
        first_capusle_layer_convolution_layer_numbers:int = 32,
        output_capsules_dimension:int = 16,
        conv1_kernel_size: int = 9,
        conv1_stride: int = 1,
        primary_caps_kernel_size: int = 9,
        primary_caps_stride: int = 2,
        # 28 is for MNIST, 48 is for FER2013
        input_image_dimension: int = 28,
        classes_number: int = 10
    ):
        super(TgnnCapNet, self).__init__()
    
        self.classes_number = classes_number

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=conv1_kernel_size, stride=conv1_stride)


        self.graph_layer = GraphLayer()
        self.tgnn_layer = TgnnLayer()
        self.tgnn_caps_layer = TgnnCapsuleLayer()
        # self.primary_capsules = CapsuleLayer(num_capsules=first_capsule_layer_dimension, num_route_nodes=-1, in_channels=256, out_channels=first_capusle_layer_convolution_layer_numbers,
                                            #  kernel_size=primary_caps_kernel_size, stride=primary_caps_stride)
        conv1_feature_map_dimension = floor( (input_image_dimension - conv1_kernel_size + conv1_stride ) / conv1_stride )
        primary_caps_feature_map_dimension = floor( (conv1_feature_map_dimension - primary_caps_kernel_size + primary_caps_stride ) / primary_caps_stride )
        self.digit_capsules = CapsuleLayer(num_capsules=self.classes_number, num_route_nodes=first_capusle_layer_convolution_layer_numbers * primary_caps_feature_map_dimension * primary_caps_feature_map_dimension, in_channels=first_capsule_layer_dimension,
                                           out_channels=output_capsules_dimension)

        self.decoder = nn.Sequential(
            nn.Linear(output_capsules_dimension * self.classes_number, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_image_dimension * input_image_dimension),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        X, A = self.graph_layer(x)
        X, A = self.tgnn_layer(X, A)
        x = self.tgnn_caps_layer(X, A)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(self.classes_number)).to(DEVICE).index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions

if __name__ == "__main__":
    _ = TgnnCapNet()