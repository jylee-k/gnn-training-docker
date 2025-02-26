import os
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

class GraphDataset:
    def __init__(self, dataset_path, task="node_classification"):
        """
        Initialize the dataset loader.
        
        :param dataset_path: Path to the dataset folder containing node_list.csv and edge_list.csv.
        :param task: Task type: "node_classification", "link_prediction", or "graph_embedding".
        """
        self.dataset_path = dataset_path
        self.task = task
        self.graph = None
        self.data = None
        self.load_graph()

    def load_graph(self):
        """Loads the graph from node and edge lists and converts it into PyTorch Geometric format."""
        node_file = os.path.join(self.dataset_path, "node_list.csv")
        edge_file = os.path.join(self.dataset_path, "edge_list.csv")

        # Load node list
        node_df = pd.read_csv(node_file)
        assert "node_id" in node_df.columns, "node_list.csv must have a 'node_id' column"

        # Node features (Optional: Check if there are feature columns)
        feature_cols = [col for col in node_df.columns if col not in ["node_id", "label"]]
        has_features = len(feature_cols) > 0
        has_labels = "label" in node_df.columns

        # Create node ID mapping (in case IDs are not 0-indexed)
        node_mapping = {node_id: i for i, node_id in enumerate(node_df["node_id"].tolist())}

        # Load edge list
        edge_df = pd.read_csv(edge_file)
        assert "source" in edge_df.columns and "target" in edge_df.columns, "edge_list.csv must have 'source' and 'target' columns"

        # Convert to adjacency format
        edge_index = torch.tensor([
            [node_mapping[src], node_mapping[tgt]] for src, tgt in zip(edge_df["source"], edge_df["target"])
        ], dtype=torch.long).T  # Transpose to match PyG format

        # Convert node features to tensor (if available)
        x = torch.tensor(node_df[feature_cols].values, dtype=torch.float) if has_features else None

        # Convert labels to tensor (if available)
        y = torch.tensor(node_df["label"].values, dtype=torch.long) if has_labels else None

        # Create PyG Data object
        self.data = Data(x=x, edge_index=edge_index, y=y)

        # Store graph as a NetworkX object for optional use
        self.graph = nx.Graph()
        self.graph.add_edges_from(zip(edge_df["source"], edge_df["target"]))

        print(f"Loaded graph with {self.data.num_nodes} nodes and {self.data.num_edges} edges")

    def get_data(self):
        """Returns the PyTorch Geometric Data object."""
        return self.data
