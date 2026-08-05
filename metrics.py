import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


from model_and_training import MLP
from Final_Project_Data_Preprocessing import test_loader

CHECKPOINT_PATH = os.path.join("output", "mlp_mnist_best.pt")
OUTPUT_DIR = "output"

def load_checkpoint(checkpoint_path, device):
    """Loads the model weights and loss history from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["train_losses"], checkpoint["val_losses"]


def plot_learning_curves(train_losses, val_losses):
    """Plots and displays the training vs. validation loss, and saves the plot."""
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
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"))
    plt.show()


def get_predictions(model, loader, device):
    """Runs test data through the model and returns predictions and true labels."""
    print("\nGetting model predictions on Test Data...\n")
    
    all_predictions = []
    all_true_labels = []

    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_labels.cpu().numpy())
    
    return all_true_labels, all_predictions


def plot_cm(all_true_labels, all_predictions):
    """Plots the confusion matrix using Seaborn and saves the plot."""
    print("\nPlotting Confusion Matrix...\n")
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title("Confusion Matrix on Test Dataset")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.show()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model, train_losses, val_losses = load_checkpoint(CHECKPOINT_PATH, device)
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find checkpoint at '{CHECKPOINT_PATH}'.")
        print("Make sure your group members have run the training script to generate it.")
        exit()


    plot_learning_curves(train_losses, val_losses)


    true_labels, predictions = get_predictions(model, test_loader, device)

    
    print("Classification Report:")
    print(classification_report(true_labels, predictions, digits=4))

   
    plot_cm(true_labels, predictions)
