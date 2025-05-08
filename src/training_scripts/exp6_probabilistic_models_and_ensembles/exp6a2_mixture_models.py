# Mixture Model (uniform and variance weighted) based on the probabilistic model

import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

import os, sys
sys.path.append('adjust/this/path')
from data.graph_instance import GIDatasetPyG

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class ModelExp6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=125, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=125, stride=2)

        # Graph Layers
        self.gcn1 = torch_geometric.nn.GCNConv(10051, 64, add_self_loops=True, normalize=True, bias=False)
        self.gcn2 = torch_geometric.nn.GCNConv(64, 64, add_self_loops=True, normalize=True, bias=False)

        # Output Layers
        self.fc_mean = nn.Linear(64, 5)  # PGA, PGV, SA03, SA10, SA30
        self.fc_variance = nn.Linear(64, 5)


    def forward(self, x1_ts, x2_static, edge_index, edge_weight):
        x = self.relu(self.conv1(x1_ts))  # out: bs x 32 x 438
        x = self.relu(self.conv2(x))  # out: bs x 64 x 157
        # flatten the cnn output
        x = torch.flatten(x, start_dim=1)  # out: bs x 10048
        # Concatenate output with static features
        x = torch.concat([x, x2_static], dim=1)  # out: bs x 10051
        # Apply graph layers
        x = self.relu(self.gcn1(x, edge_index=edge_index, edge_weight=edge_weight))  # out: bs x 64
        x = self.tanh(self.gcn2(x, edge_index=edge_index, edge_weight=edge_weight))  # out: bs x 64
        # Apply output layers
        mean = self.fc_mean(x)
        variance = self.fc_variance(x)
        # Enforce positivity of the variances. The 1e-6 is added for numerical stability
        variance = torch.log(1 + torch.exp(variance)) + 1e-6
        return mean, variance

class ModelExp6a_Mixture(nn.Module):
    def __init__(self, state_dict_paths, weighting_scheme='uniform'):
        super().__init__()
        self.num_models = len(state_dict_paths)
        self.models = nn.ModuleList([ModelExp6a() for _ in range(len(state_dict_paths))])
        self.weighting_scheme = weighting_scheme
        # load model weights
        for model, path in zip(self.models, state_dict_paths):
            # load pretrained state dict
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            # update model state dict (strict=False -> mismatches in layer names are ignored (i.e. final layer))
            model.load_state_dict(state_dict, strict=True)

    def forward(self, *inputs):
        outputs = [model(*inputs) for model in self.models]
        # `outputs` will be a list of tuples: [(mean1, var1), (mean2, var2), ...]
        means, variances = zip(*outputs)
        means, variances = torch.stack(means, dim=2), torch.stack(variances, dim=2)
        if self.weighting_scheme == 'uniform':
            weights = torch.ones_like(variances, device=device) / self.num_models
        elif self.weighting_scheme == 'weighted':
            # Calculate model weights for each node and target
            weights = F.softmax(-variances, dim=2)
        else:
            raise ValueError(f'Unknown weighting scheme: {self.weighting_scheme}')
        # compute joint weighted mean and variance
        mean = torch.sum(weights * means, dim=2) # sum across model dimension: [nodes x targets x models] -> [nodes x targets]
        variance = torch.sum(weights * (variances + means**2), dim=2) - mean**2
        return mean, variance