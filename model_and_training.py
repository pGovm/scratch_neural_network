import os
import torch
import torch.nn as nn
import torch.optim as optim

from Final_Project_Data_Preprocessing import random_seed, train_loader, validation_loader
from pathlib import Path
from tqdm import tqdm

# Building custom architecture for a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=(256, 128), num_classes=10, dropout=0.2):
        super().__init__()

        layers = []
        in_features = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) 

        return self.network(x)

# Training function
def training(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels, in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, device):
    best_val_loss = float("inf")
    epochs_no_improvement = 0
    best_state = None

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        train_loss = training(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        tqdm.write(f"Epoch {epoch + 1}/{max_epochs}    |   Training Loss: {train_loss:.4f}  |   Validation Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0
            best_state = model.state_dict()
        else:
            epochs_no_improvement+= 1

            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (Patience limit {patience} exceeded)")
                break

    model.load_state_dict(best_state)
    history = {"train_losses": train_losses, "val_losses": val_losses}

    return model, best_val_loss, history

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    common_path = BASE_DIR / "output"
    os.makedirs(common_path, exist_ok=True)

    # Standard hyperparameters used for neural network training
    learning_rate = 0.001
    epochs = 100
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, best_val_loss, history = train_model(model, train_loader, validation_loader, criterion, optimizer, epochs, patience, device)

    save_path = os.path.join(common_path, "mlp_mnist_best.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "hidden_sizes": (256, 128),
        "dropout": 0.2,
    }
    torch.save(checkpoint, save_path)

    print(f"\n Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")