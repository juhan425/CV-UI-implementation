import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchsummary import summary
import matplotlib.pyplot as plt
 
# Define transformations with and without random erasing
transform_train_with_erasing = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(),  # Add random erasing for data augmentation
 
])
 
transform_train_without_erasing = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
 
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
 
# Assuming you have the dataset in the 'training_dataset' and 'validation_dataset' folders
train_dataset_with_erasing = ImageFolder(root='Dataset_Cvdl_Hw2_Q5/dataset/training_dataset', transform=transform_train_with_erasing)
train_dataset_without_erasing = ImageFolder(root='Dataset_Cvdl_Hw2_Q5/dataset/training_dataset', transform=transform_train_without_erasing)
val_dataset = ImageFolder(root='Dataset_Cvdl_Hw2_Q5/dataset/validation_dataset', transform=transform_val)
 
# DataLoader for training and validation
train_loader_with_erasing = DataLoader(train_dataset_with_erasing, batch_size=32, shuffle=True, num_workers=4)
train_loader_without_erasing = DataLoader(train_dataset_without_erasing, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
 
# Build ResNet50 model
model = resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.Sigmoid()
)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 
# Print model summary
summary(model, (3, 224, 224))
 
# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Training and validation loop with model saving
def train_model_save_best(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_accuracy = 0.0
 
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
 
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += predicted.eq(labels.float().view(-1, 1)).sum().item()
 
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train
 
        # Validation
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
 
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().view(-1, 1))
                running_loss += loss.item()
 
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += predicted.eq(labels.float().view(-1, 1)).sum().item()
 
        val_loss = running_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
 
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()
 
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
 
    return train_losses, val_losses, train_accs, val_accs, best_model_weights, best_val_accuracy
 
# Train with random erasing
train_losses_with_erasing, val_losses_with_erasing, train_accs_with_erasing, val_accs_with_erasing, best_model_with_erasing, best_val_acc_with_erasing = train_model_save_best(
    model, train_loader_with_erasing, val_loader, criterion, optimizer, num_epochs=20
)
 
# Train without random erasing
model.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.Sigmoid()
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
train_losses_without_erasing, val_losses_without_erasing, train_accs_without_erasing, val_accs_without_erasing, best_model_without_erasing, best_val_acc_without_erasing = train_model_save_best(
    model, train_loader_without_erasing, val_loader, criterion, optimizer, num_epochs=20
)
 
# Plot bar chart for highest validation accuracy
methods = ['Without Random Erasing', 'With Random Erasing']
best_val_accs = [best_val_acc_without_erasing, best_val_acc_with_erasing]
 
plt.bar(methods, best_val_accs, color=['blue', 'blue'])
# plt.xlabel('Data Augmentation Method')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.savefig("accuracy_comparison.png")
plt.show()
 
# Save the best models
torch.save(best_model_with_erasing, 'best_model_with_erasing.pth')
torch.save(best_model_without_erasing, 'best_model_without_erasing.pth')