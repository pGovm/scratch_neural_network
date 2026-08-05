import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# Importing from our other scripts
from model_and_training import MLP
from Final_Project_Data_Preprocessing import test_loader

CHECKPOINT_PATH = os.path.join("output", "mlp_mnist_best.pt")

def load_checkpoint(checkpoint_path, device):
    """Loads the model weights and loss history from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["train_losses"], checkpoint["val_losses"]


def plot_learning_curves(train_losses, val_losses):
    """Plots and displays the training vs. validation loss."""
    epochs = len(train_losses)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
    
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.show()


def get_predictions(model, loader, device):
    """Runs test data through the model and returns predictions and true labels."""
    print("\nGetting model predictions on Test Data...\n")
    
    all_predictions = []
    all_true_labels = []

    # No gradient tracking needed for evaluation
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            
            # Get the index of the highest probability class
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_labels.cpu().numpy())
    
    return all_true_labels, all_predictions


def plot_cm(all_true_labels, all_predictions):
    """Plots the confusion matrix using Seaborn."""
    print("\nPlotting Confusion Matrix...\n")
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title("Confusion Matrix on Test Dataset")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load the checkpoint
    try:
        model, train_losses, val_losses = load_checkpoint(CHECKPOINT_PATH, device)
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find checkpoint at '{CHECKPOINT_PATH}'.")
        print("Make sure your group members have run the training script to generate it.")
        exit()

    # Step 2: Plot the training/validation loss
    plot_learning_curves(train_losses, val_losses)

    # Step 3: Get predictions
    true_labels, predictions = get_predictions(model, test_loader, device)

    # Step 4: Print the Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predictions, digits=4))

    # Step 5: Plot the Confusion Matrix
    plot_cm(true_labels, predictions)
