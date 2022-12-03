
from math import floor, sqrt
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


NUM_ROUTING_ITERATIONS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

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

def get_neighbour_adj_matrix(nodes_number):
    # Calculating row and col pos based on node number
    distance_function = lambda col1, row1, col2, row2: abs(col1 - col2) + abs(row1 - row2)

    def node_distance(node_1, node_2, n):
        '''n is matrix dimension'''
        assert n == 6
        row_1, col_1 = int(floor(node_1 / n)), int(node_1 % n)
        row_2, col_2 = int(floor(node_2 / n)), int(node_2 % n)
        return distance_function(col_1, row_1, col_2, row_2)

    def are_neighbours(node_1, node_2, total_nodes_n):
        n = int(sqrt(total_nodes_n))
        assert n == 6
        return node_distance(node_1, node_2, n) == 1

    adj = get_each_to_each_adj_matrix(nodes_number)
    
    # 2 loop search to generate the indices
    # Lame function
    indices = []
    for node_1 in range(0, nodes_number):
        for node_2 in range(0, nodes_number):
            if are_neighbours(node_1, node_2, nodes_number):
                indices.append([node_1, node_2])
    indices = torch.IntTensor(indices)

    all_edge_index = adj

    

    return adj

    # Now we know that the sum of distances between
    #  cols and row indices for all the verties must be equal 1
    # distance_function = lambda col1, row1, col2, row2: abs(col1 - col2) + abs(row1 - row2)
    # neighbour_predicate = lambda 


def get_each_to_each_adj_matrix(nodes_number):
    # Removing self loops
    mask = (torch.eye(nodes_number, nodes_number) == 0)
    adj = torch.ones(nodes_number, nodes_number) * mask
    edge_index = adj.nonzero().t()
    return edge_index

    # num_graphs =1 
    # n = 6
    # adj=torch.randint(0, 1, (num_graphs, n, n)))
    # offset, row, col = (batch.adj > 0).nonzero().t()
    # edge_weight = adj[offset, row, col]
    # row += offset * n
    # col += offset * n
    # edge_index = torch.stack([row, col], dim=0)
    # x = x.view(num_graphs * n, num_feats)
    # batch = torch.arange(0, num_graphs).view(-1, 1).repeat(1, n).view(-1)
    # edge_index = torch.randint(0, 31, (2,10))
    # edge_index=torch.eye(row,col,dtype=torch.long)
    # edge_index = edge_index.to_sparse()

    # return edge_index

class GnnCapsuleLayer(nn.Module):
 
    def __init__(self):
        super(GnnCapsuleLayer, self).__init__()
        self.gnn = SAGEConv(in_channels=8, out_channels=8)
        
    def forward(self, x):
        # x to kapsulki z warstwy primary caps

        # Zrób graf z kazdej warstwy kapsulkowej
        # Problemy techniczne:
        #   - jak pogodzic wymiary batcha z foramted edge_index
        #   - Wsparcie dla kilku warstw splotowych kapsułek, obecnie jest tylko 1
        #   - !sprawdz czy gnn sie uczy (czy jest backpropagation w warstwach przed gnn)
        
        # Kolejne kroki:
        # - zrob macierz sasiedztwa tylko dla sasiadow
        # - upewnij sie ze to dobrze dziala z batchem
        # - optymalizacja liczenia macierzy (caching)
        # - wsparcie dla wiekszej ilosci warstw kapsulkowych
        # - eksperymetny na grid ai
        # - rozwaz inne sieci gnn

        # Pomocne linki
        # https://github.com/pyg-team/pytorch_geometric/issues/1511
        nodes_number = 36
        edge_index = get_neighbour_adj_matrix(nodes_number)

        # GNN
        x = self.gnn(x,edge_index)

        return x


class TgnnCapsNet(nn.Module):
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
        super(TgnnCapsNet, self).__init__()
    
        self.classes_number = classes_number

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=conv1_kernel_size, stride=conv1_stride)
        
        self.primary_capsules = CapsuleLayer(num_capsules=first_capsule_layer_dimension, num_route_nodes=-1, in_channels=256, out_channels=first_capusle_layer_convolution_layer_numbers,
                                             kernel_size=primary_caps_kernel_size, stride=primary_caps_stride)
        
        self.gnn_capsule_layer = GnnCapsuleLayer()

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
        x = F.relu(self.conv1(x), inplace=True)
        
        x = self.primary_capsules(x)
        
        # gnn 
        x = self.gnn_capsule_layer(x)

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
    _ = TgnnCapsNet()