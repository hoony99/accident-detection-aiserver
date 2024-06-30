import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torchvision.models import densenet121, DenseNet121_Weights

# Hyperparameters
batch_size = 100
epochs = 30

# Data paths
train_path = './train'
test_path = './test'
val_path = './valid'

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model setup
device = torch.device("cuda")
torch.cuda.set_device(1) #use RTX no.1 instead of no.0
weights = DenseNet121_Weights.IMAGENET1K_V1 if torch.cuda.is_available() else DenseNet121_Weights.DEFAULT
model = densenet121(weights=weights)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # Binary classification (accident vs. no accident)
model = model.to(device)




# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)



# Helper function to calculate metrics
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    acc = accuracy_score(labels, predicted)
    recall = recall_score(labels, predicted, zero_division=0)  # Handle cases with no positive true samples
    precision = precision_score(labels, predicted, zero_division=0)  # Handle cases with no positive predictions
    f1 = f1_score(labels, predicted, zero_division=0)  # Handle cases with no positive true or predicted samples
    return acc, recall, precision, f1


# Training function
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    total_acc = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc, _, _, _ = calculate_metrics(outputs, labels)
        total_acc += acc
    avg_loss = running_loss / len(loader)
    avg_acc = total_acc / len(loader) * 100
    return avg_loss, avg_acc

# Evaluation function
def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    total_acc = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            acc, _, _, _ = calculate_metrics(outputs, labels)
            total_acc += acc
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.max(outputs, 1)[1].cpu().numpy())
    avg_loss = running_loss / len(loader)
    avg_acc = total_acc / len(loader) * 100
    return avg_loss, avg_acc, all_labels, all_predictions

# Execution
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate(model, val_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.3f}%, Val Loss: {val_loss:.5f}, Val Accuracy: {val_acc:.3f}%')

# Test the model
test_loss, test_acc, test_labels, test_predictions = evaluate(model, test_loader)
test_recall = recall_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)
print(f'Test Metrics - Accuracy: {test_acc:.3f}%, Recall: {test_recall:.3f}, Precision: {test_precision:.3f}, F1 Score: {test_f1:.3f}')

torch.save(model.state_dict(), 'densenet_model30.pth')