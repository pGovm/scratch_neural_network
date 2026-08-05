import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

random_seed = 42
torch.manual_seed(random_seed)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


print("Original training images:", len(full_train_dataset))
print("Testing images:", len(test_dataset))
print("Digit classes:", full_train_dataset.classes)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(random_seed)
)


print("\nTraining images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Testing images:", len(test_dataset))

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


images, labels = next(iter(train_loader))

print("\nImage batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 labels:", labels[:10])

def denormalize(image):
    return image * 0.3081 + 0.1307


figure = plt.figure(figsize=(10, 4))

for index in range(10):
    image = denormalize(images[index])

    plt.subplot(2, 5, index + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {labels[index].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()
