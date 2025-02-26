import argparse
import torch
from my_gnn_models import get_model
from my_datasets import get_dataset
from my_tasks import run_task

def main():
    parser = argparse.ArgumentParser(description="Train GNN in an airgapped environment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument("--task", type=str, required=True, choices=["node_classification", "link_prediction", "embedding"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=str, default="/app/output/logs.txt")
    
    args = parser.parse_args()

    # Load dataset
    dataset = get_dataset(args.dataset)
    
    # Load model
    model = get_model(args.model, dataset)
    
    # Train & Evaluate
    run_task(model, dataset, task=args.task, epochs=args.epochs, lr=args.lr, log_file=args.output)

if __name__ == "__main__":
    main()
