import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

class BaldDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Filtrar clases con menos de 2 muestras
        counts = self.annotations['type'].value_counts()
        valid_classes = counts[counts >= 2].index.tolist()
        self.annotations = self.annotations[self.annotations['type'].isin(valid_classes)].reset_index(drop=True)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.annotations['type'].unique()))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, os.path.basename(row['images']))
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            return self.__getitem__((idx + 1) % len(self))
        label = self.class_to_idx[row['type']]
        if self.transform:
            image = self.transform(image)
        return image, label

def stratified_split(dataset, test_size=0.2):
    labels = dataset.annotations['type'].tolist()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    indices = list(range(len(dataset)))
    for train_idx, val_idx in sss.split(indices, labels):
        return train_idx, val_idx

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def build_model(model_type, num_classes):
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'mobilenetv2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    elif model_type == 'custom_cnn':
        model = CustomCNN(num_classes)
    else:
        raise ValueError("Modelo no soportado")
    return model

def predict_image(image_path, model, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def load_model(model_path, model_type, num_classes, device):
    model = build_model(model_type, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    to_remove = [k for k in state_dict if 'classifier.1' in k or 'fc' in k]
    for k in to_remove:
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def train_model(csv_file, root_dir, model_type='efficientnet_b0', model_path='alopecia_model.pth', epochs=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BaldDataset(csv_file, root_dir, transform)
    class_counts = Counter(dataset.annotations['type'])
    weights = [1.0 / class_counts[label] for label in dataset.annotations['type']]

    train_idx, val_idx = stratified_split(dataset)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_labels = [dataset[i][1] for i in train_idx]
    val_labels = [dataset[i][1] for i in val_idx]

    print("Train class dist:", Counter(train_labels))
    print("Val class dist:", Counter(val_labels))

    train_sampler = WeightedRandomSampler([weights[i] for i in train_idx], len(train_idx))
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.class_to_idx)
    class_weights = torch.tensor([1.0 / class_counts[cls] for cls in sorted(class_counts)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for model_type in ['efficientnet_b0', 'mobilenetv2', 'custom_cnn']:
        print(f"\n\U0001F527 Entrenando modelo: {model_type}")
        model = build_model(model_type, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        best_acc = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    _, preds = torch.max(output, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"{model_type}_alopecia_model.pth")
                print("Modelo guardado.")

if __name__ == '__main__':
    train_model(
        csv_file="dataset/bald_people/bald_people.csv",
        root_dir="dataset/bald_people/images",
        epochs=5
    )
