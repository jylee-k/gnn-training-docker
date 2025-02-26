import torch
import torch.nn.functional as F
from torch.optim import Adam

def train(model, dataset, epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(dataset.x, dataset.edge_index)
        loss = F.cross_entropy(output, dataset.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    return model

def run_task(model, dataset, task, epochs, lr, log_file):
    with open(log_file, "w") as f:
        model = train(model, dataset, epochs, lr)
        f.write(f"Training complete. Model: {model}\n")
