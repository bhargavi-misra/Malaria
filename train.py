import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -------- CONFIG --------
DATA_DIR = "cell_images"   # change if needed
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# -------- CHECK DATA PATH --------
if not os.path.exists(DATA_DIR):
    print(f"❌ Dataset path not found: {DATA_DIR}")
    exit()

print("Loading dataset...")

# -------- TRANSFORMS --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- DATA --------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes:", dataset.classes)
print("Class to index:", dataset.class_to_idx)
print("Total images:", len(dataset))

# -------- SPLIT --------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")

# -------- LOADERS --------
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------- MODEL --------
print("Loading ResNet18...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze layers (faster training)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

print("Model ready!")

# -------- LOSS --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# -------- TRAIN --------
print("Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print every 50 batches
        if i % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)}")

    train_acc = correct / total
    print(f"\n✅ Epoch {epoch+1} Completed | Accuracy: {train_acc:.4f}\n")

# -------- SAVE --------
torch.save(model.state_dict(), "malaria_resnet18.pth")
print("🎉 Model saved as malaria_resnet18.pth")