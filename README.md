# gnn-training-docker

For creating a docker image for Graph Neural Network (GNN) training


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
Download the required libraries for the target Docker image.
```bash
mkdir -p wheelhouse
pip download -r requirements.txt -d wheelhouse --platform=manylinux2014_x86_64 --no-deps --only-binary=:all:
```

```bash
docker build -t gnn-trainer -f docker/Dockerfile .
```

Example run (bash):
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  gnn-trainer --dataset cora --model gcn --task node_classification --epochs 50 --lr 0.005 --output /app/output/results.txt
```
(PowerShell)
```PowerShell
docker run --rm --gpus all `
  -v $pwd/data:/app/data `
  -v $pwd/output:/app/output `
  gnn-trainer --dataset cora --model gcn --task node_classification --epochs 50 --lr 0.005 --output /app/output/results.txt
```

