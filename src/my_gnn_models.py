import torch.nn as nn
import torch_geometric.nn as pyg_nn

def get_model(model_name, dataset):
    if model_name == "gcn":
        return pyg_nn.GCNConv(dataset.num_features, 64)
    elif model_name == "gat":
        return pyg_nn.GATConv(dataset.num_features, 64)
    elif model_name == "sage":
        return pyg_nn.SAGEConv(dataset.num_features, 64)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
