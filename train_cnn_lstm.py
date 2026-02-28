import zipfile
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==============================
# Device Configuration
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==============================
# Dataset Extraction
# ==============================
ZIP_FILENAME = "new_approach_datasets - Copy.zip"
EXTRACT_PATH = "dataset"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Dataset Extracted.")

dataset_base = os.path.join(EXTRACT_PATH, "new_approach_datasets - Copy")

# ==============================
# File Paths
# ==============================
healthy_file = os.path.join(dataset_base, "healthy", "df_Healthy.xlsx")
fault_folder = os.path.join(dataset_base, "faultie")

label_map = {
    "healthy": 0,
    "drift": 1,
    "noise": 2,
    "offset": 3
}

# ==============================
# Load Data
# ==============================
healthy_df = pd.read_excel(healthy_file)
healthy_df["label"] = label_map["healthy"]

faulty_dfs = []

for file in os.listdir(fault_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(fault_folder, file)

        if "drift" in file:
            label = label_map["drift"]
        elif "noise" in file:
            label = label_map["noise"]
        elif "offset" in file:
            label = label_map["offset"]
        else:
            continue

        df = pd.read_csv(file_path)
        df["label"] = label
        faulty_dfs.append(df)

df_all = pd.concat([healthy_df] + faulty_dfs, ignore_index=True)

# ==============================
# Preprocessing
# ==============================
sensor_cols = [col for col in df_all.columns if col != "label"]

df_all[sensor_cols] = df_all[sensor_cols].fillna(df_all[sensor_cols].median())

for col in sensor_cols:
    df_all[col] = (df_all[col] - df_all[col].mean()) / df_all[col].std()

# ==============================
# Sliding Window Creation
# ==============================
window_size = 100
lookahead = 20
stride = 10

X, y = [], []

for i in range(0, len(df_all) - window_size - lookahead, stride):
    window = df_all.iloc[i:i + window_size]
    future = df_all.iloc[i + window_size:i + window_size + lookahead]

    X.append(window[sensor_cols].values)
    y.append(future["label"].mode()[0])

X = np.array(X)
y = np.array(y)

print("Data Shape:", X.shape, y.shape)

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train)
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ==============================
# CNN-LSTM Model
# ==============================
class CNNLSTM(nn.Module):
    def __init__(self, num_sensors, num_classes):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_sensors,
            out_channels=64,
            kernel_size=5,
            padding=2
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)

        return self.fc(h_cat)


# ==============================
# Training
# ==============================
model = CNNLSTM(num_sensors=X.shape[2], num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# ==============================
# Evaluation
# ==============================
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ==============================
# Confusion Matrix
# ==============================
rev_label_map = {v: k for k, v in label_map.items()}

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[rev_label_map[i] for i in range(len(label_map))]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Sensor Fault Prediction")
plt.show()

print("\nClassification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=[rev_label_map[i] for i in range(len(label_map))]
))
