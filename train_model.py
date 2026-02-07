import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Config
DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Dataset
class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#Load data
X = []
y = []

labels = sorted(os.listdir(DATA_DIR))

for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            path = os.path.join(label_dir, file)
            X.append(np.load(path))
            y.append(label)

X = np.array(X)        
y = np.array(y)

print("Data shape:", X.shape)
print("Labels:", set(y))

#Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

train_ds = ASLDataset(X_train, y_train)
test_ds = ASLDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

#Model (Simple LSTM)
class ASLModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=len(encoder.classes_)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = ASLModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

#Evaluation
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

acc = accuracy_score(all_true, all_preds)
print("\nTest Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=encoder.classes_))

