import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv


class GraphModel(nn.Module):
    def __init__(self, choice, num_features, hidden_dims, num_layers, output_size):
        super(GraphModel, self).__init__()
        if choice == "gcn":
            self.gnn_convs = nn.ModuleList([GCNConv(num_features, hidden_dims[0]), GCNConv(hidden_dims[0], hidden_dims[1])])
            self.linear_layers = nn.Linear(hidden_dims[0], hidden_dims[0])
            self.output_layers = nn.Linear(hidden_dims[-1], output_size)
        if choice == "cheb":
            self.gnn_convs = nn.ModuleList([ChebConv(num_features, hidden_dims[0], K=2), ChebConv(hidden_dims[0], hidden_dims[1], K=2)])
            self.linear_layers = nn.Linear(hidden_dims[0], hidden_dims[0])
            self.output_layers = nn.Linear(hidden_dims[-1], output_size)
        if choice == "sage":        
            self.gnn_convs = nn.ModuleList([SAGEConv(num_features, hidden_dims[0]), SAGEConv(hidden_dims[0], hidden_dims[1])])
            self.linear_layers = nn.Linear(hidden_dims[0], hidden_dims[0])
            self.output_layers = nn.Linear(hidden_dims[-1], output_size)
        if choice == "gat":        
            self.gnn_convs = nn.ModuleList([GATConv(num_features, hidden_dims[0]), GATConv(hidden_dims[0], hidden_dims[1])])
            self.linear_layers = nn.Linear(hidden_dims[0], hidden_dims[0])
            self.output_layers = nn.Linear(hidden_dims[-1], output_size)

    """def forward(self, dataloader):  # subject-wise
        all_outputs = []
        for idx, (gnn_conv, linear_layer, data) in enumerate(zip(self.gnn_convs, self.linear_layers, dataloader)):
            x = gnn_conv(data.x, data.edge_index)
            x = torch.tanh(x)
            x = linear_layer(x)
            x = self.output_layers(x)
            all_outputs.append(x)
        stacked_embeddings = torch.stack(all_outputs, dim=0)
        out = stacked_embeddings[0]
        out = F.adaptive_max_pool1d(out.t().unsqueeze(0), output_size=1).squeeze(0).t()
        out = F.softmax(out, dim=1)
        return out"""
    
    def forward(self, dataloader):  # subject-wise
        all_outputs = []
        for data in dataloader:
            edge_index = data.edge_index
            x = self.gnn_convs[0](data.x, edge_index)
            x = torch.tanh(x)
            x = self.linear_layers(x)
            x = self.gnn_convs[1](x, edge_index)
            x = torch.tanh(x)
            x = self.output_layers(x)
            all_outputs.append(x)
        stacked_embeddings = torch.stack(all_outputs, dim=0)
        out = stacked_embeddings[0]
        out = F.adaptive_max_pool1d(out.t().unsqueeze(0), output_size=1).squeeze(0).t()
        #out = F.softmax(out, dim=1)
        return out