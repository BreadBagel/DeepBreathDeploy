import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Custom Dataset ===
class CoughImageDataset(Dataset):
    def __init__(self, pneum_dir, bronch_dir, transform=None):
        self.pneum_dir = pneum_dir
        self.bronch_dir = bronch_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for img in os.listdir(pneum_dir):
            self.image_paths.append(os.path.join(pneum_dir, img))
            self.labels.append(0)  # 0 = Pneumonia

        for img in os.listdir(bronch_dir):
            self.image_paths.append(os.path.join(bronch_dir, img))
            self.labels.append(1)  # 1 = Bronchitis

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# === ResNet34 Model Loader ===
def get_resnet_model(num_classes=2):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# === Set your data directories ===
PNEUM_DIR = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'   # Update this
BRONCH_DIR = r'C:\Users\User\Downloads\organized_cough_dataset\Negative' # Update this

# === Configuration ===
num_epochs = 10
batch_size = 8
num_folds = 5
learning_rate = 0.0003

# === Dataset & Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = CoughImageDataset(PNEUM_DIR, BRONCH_DIR, transform=transform)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# === Cross-validation Training ===
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f'\n=== Training Fold {fold+1}/{num_folds} ===')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = get_resnet_model(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Validation accuracy: {accuracy:.4f}")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f'resnet34_best_fold{fold}.pth')

    fold_accuracies.append(best_acc)

print(f'\n=== Cross-validation Complete ===')
print(f'Average accuracy: {np.mean(fold_accuracies):.4f}')
