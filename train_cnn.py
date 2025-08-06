import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# ===== Step 1: Define Paths =====
data_dir = 'data/train'                    # Folder should contain subfolders per class
model_path = 'model/coconut_cnn.pth'       # Where the model will be saved
os.makedirs('model', exist_ok=True)        # Create model folder if it doesn't exist

# ===== Step 2: Choose Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# ===== Step 3: Define Image Transforms =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize all images to 224x224
    transforms.ToTensor(),                 # Convert to PyTorch tensors
])

# ===== Step 4: Load Dataset =====
print("📁 Loading dataset...")
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"✅ Loaded {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

# ===== Step 5: Define CNN Model (ResNet18) =====
print("🧠 Initializing ResNet18...")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

# ===== Step 6: Set Loss and Optimizer =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== Step 7: Train the Model =====
print("🚀 Starting training...")
for epoch in range(150):  # 🔁 You can increase this for better accuracy
    total_loss = 0
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f" Epoch [{epoch+1}/150], Loss: {total_loss:.4f}")

# ===== Step 8: Save the Trained Model =====
print(f"💾 Saving model to {model_path}...")
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': dataset.classes
}, model_path)

print("✅ Training complete! Model saved.") 