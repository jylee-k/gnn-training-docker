# gnn-training-docker

For creating a Python docker image for Graph Neural Network (GNN) training

Python 3.11


### Image configuration

```
/app
│── src
│   ├── train.py               # Main training script
│   ├── my_gnn_models.py       # Model definitions (GCN, GAT, etc.)
│   ├── my_datasets.py         # Dataset loading utilities
│   ├── my_tasks.py            # Task-specific execution
│── wheelhouse                 # Pre-downloaded Python packages for offline install
│── requirements.txt           # Dependency list
│── output                     # Directory where logs & results are stored
│   ├── results.txt            # Training logs & results
```

### For offline deployment
To download the wheels for offline (e.g. airgapped environment) deployment:
```bash
mkdir -p wheelhouse
pip download -r requirements.txt -d wheelhouse
```

```
docker build -t gnn-trainer -f docker/Dockerfile .
```

Example run:
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  gnn-trainer --dataset cora --model gcn --task node_classification --epochs 50 --lr 0.005 --output /app/output/results.txt
```

